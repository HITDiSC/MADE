import json
import os
import re
import shlex
import subprocess
import collections
from abc import ABC, abstractmethod
import yaml
import ast
import time
import sys
from pathlib import Path
from typing import Union, List, Dict, Callable, Any, Set, Optional
from backend.query import query, json_query
from agenttool.base_phase import BasePhase
from agenttool.tool import build_tree, linux_command, locate_path, inspect_running
from agenttool.gpu_profile import resolve_gpu_profile, compute_cuda_torch_hint

file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)

try:
    with open(os.path.join(project_path, "config/global.yaml"), "r") as f:
        global_config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found.")
except yaml.YAMLError as exc:
    raise yaml.YAMLError(f"Error in configuration file: {exc}")

docker_setting = global_config.get("docker_setting") or {}
container_name = docker_setting.get("container_name")
image_name = docker_setting.get("image_name")
port_mapping = docker_setting.get("port_mapping")
use_gpu = docker_setting.get("use_gpu")
work_dir = docker_setting.get("work_dir") or "/workspace"

REQ_SPLIT_RE = re.compile(r"(===|==|!=|>=|<=|~=|>|<)")

# Packages that almost never belong in an inference-only container image.
# Dropped unconditionally from merge_dependencies() at the end, regardless of
# whether they came in via requirements*.txt or via imports from the repo.
#
# Entries are compared via _canonical_pip_name() which matches PEP 503:
# lowercase, collapse runs of [-_.] to a single '-', strip extras like [all].
# So writing "pytest" here also matches "Pytest", "py_test", "pytest[mock]".
#
# Keep this list high-confidence. Borderline packages (accelerate,
# pytorch-lightning, deepspeed, onnx, transformers) are NOT denylisted
# because they have legitimate inference uses.
INFERENCE_DENYLIST: Set[str] = {
    # Experiment tracking / logging
    "wandb", "tensorboard", "tensorboardx", "mlflow",
    "neptune", "neptune-client", "comet-ml", "aim", "clearml",
    # Test runners / fixtures
    "pytest", "pytest-cov", "pytest-xdist", "pytest-mock",
    "pytest-asyncio", "pytest-timeout", "pytest-benchmark",
    "coverage", "hypothesis", "tox", "nox",
    # Lint / format / type-check
    "black", "isort", "flake8", "mypy", "ruff", "pylint",
    "pyflakes", "pycodestyle", "pre-commit", "autopep8", "yapf",
    # Docs
    "sphinx", "sphinx-rtd-theme", "sphinx-autodoc-typehints",
    "mkdocs", "mkdocs-material", "myst-parser", "furo", "recommonmark",
    # Notebook / interactive
    "jupyter", "jupyterlab", "notebook", "ipykernel", "ipywidgets",
    "jupytext", "nbconvert", "jupyter-client",
    # Hyperparameter search (training-only)
    "optuna", "hyperopt", "bayesian-optimization", "ray[tune]",
    # Training-only distributed frameworks
    "horovod", "bagua",
    # Python packaging/build plumbing (ships with python or handled by base image)
    "setuptools", "wheel", "build", "twine", "pip",
}


def _canonical_pip_name(name: str) -> str:
    """PEP 503 style normalization: lowercase, [-_.]+ -> '-', drop extras.

    'Pytest-Cov' / 'pytest_cov' / 'pytest.cov' / 'pytest-cov[all]' all normalize
    to 'pytest-cov'.
    """
    if not name:
        return ""
    base = name.split("[", 1)[0].strip().lower()
    return re.sub(r"[-_.]+", "-", base)


_INFERENCE_DENYLIST_CANONICAL: Set[str] = {_canonical_pip_name(n) for n in INFERENCE_DENYLIST}

# Directories whose .py files are almost certainly NOT on the inference code
# path. Skipped when collecting imports so dev/test/training-only packages
# don't leak into the Dockerfile. `scripts/` and `tools/` are intentionally
# kept OUT of this list because some repos put their real inference entry
# point (e.g. scripts/predict.py) there - filtering them would under-collect.
_IMPORT_SCAN_SKIP_DIRS: Set[str] = {
    # Build / env noise (previous behavior)
    ".git", "venv", ".venv", "__pycache__", "build", "dist",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
    # Test surfaces
    "tests", "test", "testing", "__tests__",
    # Examples / demos / benchmarks
    "examples", "example", "demo", "demos",
    "benchmark", "benchmarks", "bench",
    # Documentation
    "docs", "doc", "documentation", "site",
    # Notebooks
    "notebooks", "notebook",
    # Training-only
    "training", "trainer", "experiments", "experiment", "exp",
}

# File-stem substring patterns that mark a requirements file as
# training/dev/test/docs/build only. Used by find_requirement_files.
_REQUIREMENTS_NON_INFERENCE_PATTERNS: tuple = (
    "dev", "develop", "test", "testing",
    "train", "training",
    "lint", "format", "style",
    "doc", "docs",
    "build", "ci", "release",
    "benchmark", "bench",
    "notebook",
)

# import name -> pip distribution name. Common cases where they differ.
IMPORT_TO_PIP: Dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyOpenSSL",
    "dateutil": "python-dateutil",
    "google": "google-api-python-client",
    "torch_geometric": "torch-geometric",
    "pytorch_lightning": "pytorch-lightning",
    "lightning": "lightning",
    "MySQLdb": "mysqlclient",
    "psycopg2": "psycopg2-binary",
    "magic": "python-magic",
    "Levenshtein": "python-Levenshtein",
    "tensorflow_text": "tensorflow-text",
    "wandb": "wandb",
    "tb": "tensorboard",
}

# Filenames that look like a Dockerfile leftover/backup, not a real Dockerfile.
_DOCKERFILE_REJECT_SUFFIXES = (".txt", ".bak", ".md", ".old", ".sample", ".tmpl", ".template")


def _is_dockerfile_name(name: str) -> bool:
    """Whether a filename should be treated as a Dockerfile candidate.

    Accepts: 'Dockerfile', 'Dockerfile.dev', 'Dockerfile-gpu', 'gpu.Dockerfile'.
    Rejects: '.dockerignore', 'dockerfilebackup.txt', 'Dockerfile.bak', etc.
    """
    if not name or name.startswith("."):
        return False
    lower = name.lower()
    if lower == "dockerfile":
        return True
    if lower.endswith(_DOCKERFILE_REJECT_SUFFIXES):
        return False
    if lower.startswith("dockerfile.") or lower.startswith("dockerfile-"):
        return True
    if lower.endswith(".dockerfile"):
        return True
    return False

class DockerSetUp(BasePhase):
    name: str = "DockerSetUp"
    description: str = "Set up the Docker environment."
    goal: str = "get the Docker image"
    tools_schemas: List[Dict[str, any]] = [
        {"name": "find_local_dockerfile", "description": "Look for an existing Dockerfile inside the repository. Whether or not a local Dockerfile is found, ALWAYS use dockerfile_generation next to synthesize an inference-optimized Dockerfile adapted to the current machine. A found local Dockerfile will be automatically used as reference context by dockerfile_generation — do NOT pass it directly to docker_build.", "args": {"repo_root": Path}},
        {"name": "dockerfile_generation", "description": "Generate the Dockerfile. If there is no local dockerfile, use this tool to generate the Dockerfile. hint is an optional free-form string where you can describe any extra context, diagnosis, or constraints you want the Dockerfile generator to consider (e.g. 'this project requires CUDA 11.8 and Python 3.9', 'the repo uses a custom C extension that needs gcc and cmake'). Always provide a hint when you have useful context — it improves generation accuracy.", "args": {"repo_root": Path, "hint": str}},
        {"name": "docker_build", "description": "Build the Docker image. Using this tool to build the Docker image.", "args": {"repo_root": Path, "dockerfile_path": Path}},
        {"name": "docker_image_inspect", "description": "Inspect a built Docker image and return its metadata, including config, entrypoint, command, and working directory. Use this after docker_build to verify the image before running a container.", "args": {"image_name": str}},
        {"name": "verify_gpu_runtime", "description": "Run a throwaway container from the just-built image and probe its torch/CUDA stack. Call this AFTER docker_build and BEFORE docker_run whenever GPU is enabled in the gpu_profile. Pass dockerfile_path (the path returned by dockerfile_generation) so the probe can detect the exact Python interpreter used to install torch and avoid false ModuleNotFoundError from interpreter mismatch. On failure, re-run dockerfile_generation with the new evidence instead of proceeding to docker_run. When GPU is not enabled, this tool reports a skipped status and is safe to omit.", "args": {"image_name": str, "dockerfile_path": Path}},
        {"name": "docker_run", "description": "(COMPULSORY)Run the Docker container. Use this AFTER docker_build, and AFTER verify_gpu_runtime succeeds when GPU is enabled. If verify_gpu_runtime reported gpu_runtime_mismatch, do NOT proceed to docker_run - regenerate the Dockerfile first.", "args": {"repo_root": Path, "image_name": str}},
        {"name": "read_file", "description": "Read the content of a file and return it. Use this to inspect build logs, Dockerfiles, or other files when you need to diagnose errors. Returns the file content (truncated to the last 200 lines for large files).", "args": {"path": str}},
    ]
    allowed_parallel_phases: List[str] = ["weightresolve"]

    def __init__(self) -> None:
        super().__init__()
        self.backend = "gr"
        self.gpu_profile = resolve_gpu_profile()
        self.tools = {
            "find_local_dockerfile": self.find_local_dockerfile,
            "dockerfile_generation": self.dockerfile_generation,
            "docker_build": self.docker_build,
            "docker_image_inspect": self.docker_image_inspect,
            "verify_gpu_runtime": self.verify_gpu_runtime,
            "docker_run": self.docker_run,
            "read_file": self.read_file,
        }
        prompt_path = os.path.join(project_path, "prompts")
        local_dockerfile_prompt_filepath = os.path.join(prompt_path, "local_dockerfile_prompt.json")
        with open(local_dockerfile_prompt_filepath, "r", encoding="utf-8") as f:
            self.local_dockerfile_prompt = json.load(f)
        docker_build_prompt_filepath = os.path.join(prompt_path, "dockerfile_prompt.json")
        with open(docker_build_prompt_filepath, "r", encoding="utf-8") as f:
            self.docker_build_prompt = json.load(f)

        # Phase-instance feedback channel between docker_build and
        # dockerfile_generation. When docker_build fails it stashes the error
        # tail here; when dockerfile_generation runs again within the same
        # phase execution it reads the previous Dockerfile from disk + this
        # error and feeds both back to the LLM so it can learn from the
        # mistake instead of re-emitting the same broken Dockerfile.
        #
        # Lifecycle within one execute_phase() call:
        #   (1) dockerfile_generation #1 -> writes Dockerfile
        #   (2) docker_build          #1 -> fails, stashes error here
        #   (3) dockerfile_generation #2 -> reads prev Dockerfile + self
        #                                    ._last_build_error, regenerates
        #   (4) docker_build          #2 -> success clears self._last_build_error
        #                                    OR failure overwrites it
        #
        # phase is re-instantiated per execute_phase() call (main.py:82:
        # `phase = phase()`), so this state never leaks across phases.
        self._last_build_error: str = ""

        # Companion channel for verify_gpu_runtime. A successful docker_build
        # does NOT mean the image's torch/CUDA stack matches the host driver,
        # so verify_gpu_runtime can fail after a clean build. We stash that
        # failure here and feed it into the next dockerfile_generation call
        # so the LLM can pick a compatible torch wheel / base image instead
        # of regenerating the same mismatched Dockerfile. Cleared on a
        # successful verify_gpu_runtime, and on a successful docker_build
        # (the new image invalidates any prior runtime-mismatch evidence).
        self._last_gpu_runtime_error: str = ""

        # Attempt counter: how many times dockerfile_generation has fired in
        # this phase execution. Fed into the LLM prompt so rule 24 can permit
        # a CPU-torch fallback after repeated torch/CUDA failures, instead of
        # looping on the same incompatible wheel forever. The counter is
        # phase-instance-scoped (main.py re-instantiates the phase per
        # execute_phase), so it naturally resets across phases.
        self._dockerfile_generation_attempts: int = 0

    def boundary_tools(self, tool_name: str) -> bool:
        boundary_tools_list = ["docker_run"]
        if tool_name in boundary_tools_list:
            return True
        else:
            return False

    def tool_arguments(self, tool_name: str) -> Dict[str, any]:
        tool_arguments_dict = {
            "find_local_dockerfile": {"repo_root": Path},
            "dockerfile_generation": {"repo_root": Path, "hint": str},
            "docker_build": {"repo_root": Path, "dockerfile_path": Path},
            "docker_image_inspect": {"image_name": str},
            "verify_gpu_runtime": {"image_name": str, "dockerfile_path": Path},
            "docker_run": {"repo_root": Path, "image_name": str},
            "read_file": {"path": str},
        }
        return tool_arguments_dict[tool_name]

    def assemble_local_dockerfile_prompt(self, dockerfile_candidates: List[dict]) -> str:
        prompt_parts = []

        role_prompt = "\n".join(self.local_dockerfile_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        dockerfile_candidates_prompt = "\n".join(self.local_dockerfile_prompt['dockerfile_candidates']).format(
            dockerfile_candidates=json.dumps(dockerfile_candidates)
        )
        prompt_parts.append("\n=== DOCKERFILE CANDIDATES ===")
        prompt_parts.append(dockerfile_candidates_prompt)

        output_format_prompt = json.dumps(self.local_dockerfile_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        return "\n".join(prompt_parts)

    def find_dockerfile_files(self, tree):
        results = []

        def dfs(node, current_path=""):
            path = os.path.join(current_path, node["name"])

            if node["type"] == "file" and _is_dockerfile_name(node["name"]):
                results.append({
                    "name": node["name"],
                    "path": path,
                    "size": node["size"]
                })

            if node["type"] == "dir":
                children = node.get("children", [])
                if isinstance(children, list):
                    for child in children:
                        dfs(child, path)

        dfs(tree)
        return results

    def find_local_dockerfile(self, repo_root: Path) -> List[Dict[str, Any]]:
        # All three "no usable Dockerfile found" paths below return
        # storage="temporary" (not "error") and variable_name
        # "local_dockerfile_absent". These are branching signals telling the
        # EM to proceed to dockerfile_generation, NOT real tool failures.
        # Reporting them as errors would make PhaseSummarizer mark the phase
        # as partial and cause the PM to needlessly retry a phase that
        # actually succeeded.
        repo_tree = build_tree(repo_root)
        dockerfile_candidates = self.find_dockerfile_files(repo_tree)
        if dockerfile_candidates == []:
            return [{
                "value": "There is no dockerfile, create a dockerfile.",
                "storage": "temporary",
                "variable_name": "local_dockerfile_absent"
            }]
        else:
            prompt = self.assemble_local_dockerfile_prompt(dockerfile_candidates)
            response = json_query(prompt, "local_dockerfile_find", self.backend)
            if isinstance(response, str):
                response = json.loads(response)
            find_dockerfile = response['find_dockerfile']
            if isinstance(find_dockerfile, str):
                find_dockerfile = find_dockerfile.strip().lower() == "true"
            if find_dockerfile:
                dockerfile_path = response['dockerfile_path']
                found, dockerfile_path, node_type = locate_path(repo_root, dockerfile_path)
                if found:
                    return [{
                        "value": str(dockerfile_path),
                        "storage": "temporary",
                        "variable_name": "local_dockerfile_path"
                    },
                    {
                        "value": (
                            f"Found local Dockerfile at '{dockerfile_path}'. "
                            "It will be used as reference context automatically. "
                            "Now use dockerfile_generation to create an inference-optimized "
                            "Dockerfile adapted to the current machine's GPU/CUDA environment."
                        ),
                        "storage": "temporary",
                        "variable_name": "local_dockerfile_found_as_reference"
                    }]
                else:
                    return [{
                        "value": "Dockerfile candidate existed in tree but could not be located on disk; create a dockerfile.",
                        "storage": "temporary",
                        "variable_name": "local_dockerfile_absent"
                    }]
            else:
                return [{
                    "value": "There is no local dockerfile, generate a dockerfile.",
                    "storage": "temporary",
                    "variable_name": "local_dockerfile_absent"
                }]

    def merge_dependencies(self, repo_root: Path) -> List[str]:
        """
        Build the Python dependency list fed to dockerfile_generation.

        Pipeline:
        1. Load pinned deps from non-dev requirements*.txt files (version
           specs preserved; requirements.txt wins on collisions).
        2. Add import-inferred deps as a fallback (no version; import name
           first mapped through IMPORT_TO_PIP to the real pip dist name).
        3. Drop anything in INFERENCE_DENYLIST (experiment tracking,
           dev/test tooling, docs, notebook, HPO, build plumbing) so the
           final image stays inference-focused.

        Returns: ordered list[str] of pip requirement lines.
        """

        final: Dict[str, str] = {}

        # 1. requirements*.txt (filtered to non-dev variants)
        req_files = self.find_requirement_files(repo_root)
        req_files = sorted(
            req_files,
            key=lambda p: (p.name != "requirements.txt", p.name)
        )
        for req_path in req_files:
            req_pkgs = self.read_requirements(req_path)
            for name, spec in req_pkgs.items():
                if name not in final:
                    final[name] = spec

        # 2. imports-inferred fallback
        import_pkgs = self.extract_import_pkgs(repo_root)
        for name in sorted(import_pkgs):
            pip_name = IMPORT_TO_PIP.get(name, name)
            if pip_name not in final and name not in final:
                final[pip_name] = pip_name

        # 3. Apply INFERENCE_DENYLIST as a final pass. Compare via canonical
        # names so Pytest / pytest_cov / pytest-cov[all] all collapse to the
        # same key. This is deterministic and audited via the drop log below.
        dropped: List[str] = []
        kept: Dict[str, str] = {}
        for name, spec in final.items():
            if _canonical_pip_name(name) in _INFERENCE_DENYLIST_CANONICAL:
                dropped.append(spec)
                continue
            kept[name] = spec

        if dropped:
            print(
                f"[DockerSetUp] merge_dependencies: dropped {len(dropped)} "
                f"non-inference packages via INFERENCE_DENYLIST: {dropped}"
            )
        print(
            f"[DockerSetUp] merge_dependencies: kept {len(kept)} packages "
            f"for dockerfile_generation."
        )

        return list(kept.values())

    def _collect_local_modules(self, repo_root: Path) -> Set[str]:
        """
        Collect the repo's local module names (for the top-level import section).

        Covers:
        - top-level .py files at the repo root (`foo.py` -> `foo`)
        - subdirectories under the repo root containing any .py file (including subpackage names in a src layout)
        - package directory names (at any level) that contain an __init__.py
        """
        local_modules: Set[str] = set()

        for entry in repo_root.iterdir():
            if entry.is_file() and entry.suffix == ".py":
                local_modules.add(entry.stem)
                continue
            if entry.is_dir():
                if entry.name.startswith(".") or entry.name in {"__pycache__", "venv", ".venv", "node_modules"}:
                    continue
                # any single .py marks this as a local package
                if any(True for _ in entry.rglob("*.py")):
                    local_modules.add(entry.name)

        # directory names containing an __init__.py (any level) count too
        for init_file in repo_root.rglob("__init__.py"):
            parent = init_file.parent
            if parent == repo_root:
                continue
            local_modules.add(parent.name)

        return local_modules

    def read_requirements(self, req_path: Path):
        """
        Read a single requirements file.
        Returns dict: {package_name: full dependency string}
        """
        pkgs = {}

        if not req_path.exists():
            return pkgs

        lines = req_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        for line in lines:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # skip editable / git / local path
            if line.startswith(("-e ", "git+", "./", "../")):
                continue

            # extract the package name (without mangling valid names)
            parts = REQ_SPLIT_RE.split(line, maxsplit=1)
            pkg_name = parts[0].strip()

            if pkg_name:
                pkgs[pkg_name] = line  # keep the original dependency spec

        return pkgs

    def extract_import_pkgs(self, folder: Path):
        """
        Extract imported packages from a Python file (names only, no versions).
        Returns set: {"torch", "numpy", "tqdm", ...}
        """
        imports = set()
        try:
            stdlib_modules = set(sys.stdlib_module_names)  # type: ignore[attr-defined]
        except Exception:
            stdlib_modules = {
                "abc", "argparse", "ast", "asyncio", "base64", "collections", "concurrent",
                "contextlib", "csv", "ctypes", "datetime", "decimal", "functools", "glob",
                "hashlib", "heapq", "http", "importlib", "inspect", "io", "itertools",
                "json", "logging", "math", "multiprocessing", "os", "pathlib", "pickle",
                "queue", "random", "re", "shutil", "signal", "socket", "sqlite3", "statistics",
                "string", "subprocess", "sys", "tempfile", "threading", "time", "typing",
                "unittest", "urllib", "uuid", "warnings", "xml", "zipfile"
            }

        # scan once to build the local-module set, avoiding a filesystem lookup per import,
        # while correctly recognizing src-layout / nested subpackages.
        local_modules = self._collect_local_modules(folder)

        # Use the module-level _IMPORT_SCAN_SKIP_DIRS so find_requirement_files
        # and extract_import_pkgs agree on what "non-inference" means. Skipping
        # tests/ examples/ docs/ notebooks/ training/ here is what prevents
        # pytest / sphinx / wandb / jupyter from being added to the Dockerfile
        # just because some peripheral script imported them.
        skip_dir_names = _IMPORT_SCAN_SKIP_DIRS

        for py_file in folder.rglob("*.py"):
            if not py_file.is_file():
                continue
            # Skip anything nested under an ignored directory. Any path segment
            # matching gets the whole file dropped, so "examples/foo/bar.py"
            # is skipped via the "examples" segment check.
            if any(part in skip_dir_names for part in py_file.relative_to(folder).parts):
                continue

            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except Exception:
                continue

            for node in ast.walk(tree):

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name.split(".")[0]
                        if name in stdlib_modules or name in local_modules:
                            continue
                        imports.add(name)

                elif isinstance(node, ast.ImportFrom):
                    # treat all relative imports as local
                    if node.level and node.level > 0:
                        continue
                    if node.module:
                        name = node.module.split(".")[0]
                        if name in stdlib_modules or name in local_modules:
                            continue
                        imports.add(name)
        return imports

    def find_requirement_files(self, repo_root: Path) -> List[Path]:
        """
        Locate inference-relevant requirements files at the repo root.

        Rules:
        - Must be a top-level *.txt with 'require' in the filename
        - Exclude variants whose stem contains dev/test/train/doc/lint/build
          markers (see _REQUIREMENTS_NON_INFERENCE_PATTERNS)
        - If filtering removes everything (e.g. the repo only ships
          `requirements-train.txt`), fall back to the unfiltered list
          because "one bad list is better than zero pinned versions"
        """
        all_candidates: List[Path] = []
        filtered: List[Path] = []

        for file in repo_root.iterdir():
            if not file.is_file():
                continue

            suffix = file.suffix.lower()
            name = file.name.lower()

            if suffix != ".txt" or "require" not in name:
                continue

            all_candidates.append(file)

            stem = file.stem.lower()  # "requirements-dev"
            if any(pat in stem for pat in _REQUIREMENTS_NON_INFERENCE_PATTERNS):
                print(
                    f"[DockerSetUp] find_requirement_files: skipping {file.name} "
                    f"(stem matches non-inference pattern)"
                )
                continue
            filtered.append(file)

        if filtered:
            return sorted(filtered)

        if all_candidates:
            print(
                "[DockerSetUp] find_requirement_files: all candidates look "
                "non-inference, falling back to unfiltered set to avoid an "
                "empty dependency baseline."
            )
        return sorted(all_candidates)

    def detect_runtime_files(self, repo_root: Path) -> List[Dict[str, Any]]:
        patterns = {
            "requirements_files": ["requirements*.txt"],
            "pyproject_files": ["pyproject.toml"],
            "setup_files": ["setup.py", "setup.cfg"],
            "environment_files": ["environment.yml", "environment.yaml"],
            "pipfile_files": ["Pipfile", "Pipfile.lock"],
        }
        runtime_files = {key: [] for key in patterns}

        for category, globs in patterns.items():
            found_paths = []
            for pattern in globs:
                found_paths.extend(
                    str(path.relative_to(repo_root)).replace("\\", "/")
                    for path in repo_root.rglob(pattern)
                    if path.is_file()
                )
            runtime_files[category] = sorted(set(found_paths))

        return [{
            "value": runtime_files,
            "storage": "temporary",
            "variable_name": "runtime_files"
        }]

    def _gpu_profile_for_prompt(self) -> Dict[str, Any]:
        """Projection of self.gpu_profile that ONLY exposes the selected GPUs
        (plus host driver / CUDA ceiling, which rules 11-14 require to pick a
        compatible base image + torch wheel) and attaches a deterministic
        cuda_torch_hint derived from the selected GPU architecture.

        Rationale: the LLM does not need to know about unselected GPUs on the
        host - feeding the full inventory invites it to reason about cards it
        should ignore. Field 'available_gpu_count' is dropped for that
        reason. The hint narrows the LLM's job to Dockerfile syntax when the
        GPU classifies cleanly; when it doesn't (unknown GPU model, mixed
        conflicts), hint.resolved is False and the LLM falls back to the
        generic ceiling-based rules in the prompt.
        """
        src = self.gpu_profile or {}
        keep_keys = (
            # Intent
            "use_gpu", "gpu_enabled",
            # Host-level compatibility ceiling (shared by all selected GPUs)
            "driver_version", "cuda_runtime_ceiling",
            # Selected GPUs only
            "selected_gpu_indices", "selected_gpus", "gpu_count",
            "per_gpu_memory_gb", "min_per_gpu_memory_gb",
            "max_per_gpu_memory_gb", "total_selected_memory_gb",
        )
        projected = {k: src.get(k) for k in keep_keys if k in src}
        # Only compute a hint when GPU is actually enabled for this deploy.
        # With GPU disabled, CUDA/torch choices are not relevant and an
        # "unknown" hint would just add noise.
        if src.get("gpu_enabled"):
            projected["cuda_torch_hint"] = compute_cuda_torch_hint(
                src.get("selected_gpus") or [],
                src.get("cuda_runtime_ceiling"),
            )
        return projected

    def assemble_docker_build_prompt(
        self,
        imports: List[str],
        runtime_files: Dict[str, List[str]],
        gpu_profile: Dict[str, Any],
        previous_attempt: Optional[Dict[str, str]] = None,
        attempt_number: int = 1,
        hint: str = "",
    ) -> str:
        prompt_parts = []

        role_prompt = "\n".join(self.docker_build_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        rules_prompt = "\n".join(self.docker_build_prompt['rules'])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)

        imports_prompt = "\n".join(self.docker_build_prompt['imports']).format(
            imports=imports
        )
        prompt_parts.append("\n=== IMPORTS ===")
        prompt_parts.append(imports_prompt)

        runtime_files_prompt = "\n".join(self.docker_build_prompt['runtime_files']).format(
            runtime_files=json.dumps(runtime_files, ensure_ascii=False)
        )
        prompt_parts.append("\n=== RUNTIME FILES ===")
        prompt_parts.append(runtime_files_prompt)

        gpu_profile_prompt = "\n".join(self.docker_build_prompt['gpu_profile']).format(
            gpu_profile=json.dumps(gpu_profile, ensure_ascii=False)
        )
        prompt_parts.append("\n=== GPU PROFILE ===")
        prompt_parts.append(gpu_profile_prompt)

        # Attempt counter: lets the LLM apply rule 24 (CPU-torch fallback
        # authorized at attempt >= 3 after repeated torch/CUDA failures)
        # without us having to pattern-match the error ourselves.
        prompt_parts.append("\n=== ATTEMPT CONTEXT ===")
        prompt_parts.append(
            f"attempt_number: {attempt_number} (counts how many times "
            f"dockerfile_generation has run in this phase execution; starts at 1)"
        )

        # Previous-version section: the single most recent Dockerfile known
        # for this repo (the repo's original on the first call, our own last
        # generation afterwards), paired with the build error it produced if
        # any. We keep only ONE previous Dockerfile here because the stashed
        # error corresponds to that specific Dockerfile; including older
        # versions would break the 1:1 mapping between Dockerfile and error.
        if previous_attempt is not None:
            build_error = previous_attempt.get("build_error", "")
            gpu_error = previous_attempt.get("gpu_runtime_error", "")
            has_error = bool(build_error) or bool(gpu_error)
            prev_path = previous_attempt.get("dockerfile_path", "") or "Dockerfile"

            if has_error:
                # Compose a header that reflects which stage(s) failed. A
                # gpu_runtime failure typically happens AFTER a clean build,
                # so "FAILED to build" alone would be misleading.
                if build_error and gpu_error:
                    stage_label = "failed to build AND failed GPU runtime verification"
                elif build_error:
                    stage_label = "FAILED to build"
                else:
                    stage_label = "built successfully but FAILED GPU runtime verification"

                header = "\n=== PREVIOUS DOCKERFILE ATTEMPT FAILED ==="
                body = (
                    f"The previous Dockerfile (at '{prev_path}') {stage_label}. "
                    "Its full content and the tail(s) of the relevant error(s) are below.\n\n"
                    "Your job now is:\n"
                    "1. Read the previous Dockerfile AND every error section carefully.\n"
                    "2. Identify the single root cause (wrong index, missing apt "
                    "package, torch/cuda mismatch incompatible with the host driver, "
                    "nonexistent pin, etc). For a GPU runtime error the fix is almost "
                    "always a torch wheel or base image that matches the host "
                    "driver_version / cuda_runtime_ceiling reported in GPU PROFILE.\n"
                    "3. Produce a NEW Dockerfile that fixes exactly that root cause. "
                    "Keep everything else identical unless another rule forces a change.\n"
                    "4. Do NOT re-emit the same commands that caused the original "
                    "failure. If the same pip install line or the same base image or "
                    "the same --index-url flag appeared in the previous Dockerfile "
                    "and was the root cause, it MUST be different this time.\n"
                    "5. Never 'fix' the error by removing a dependency the code "
                    "actually needs - find a working install recipe instead."
                )
            else:
                header = "\n=== PREVIOUS DOCKERFILE (REFERENCE) ==="
                body = (
                    f"The repository already contained a Dockerfile at '{prev_path}'. "
                    "Treat it as a strong reference:\n"
                    "1. Prefer reusing its base image, system packages, entrypoint, "
                    "and working directory unless they contradict the IMPORTS / "
                    "RUNTIME FILES / GPU PROFILE above.\n"
                    "2. Adapt only what is necessary to satisfy the constraints "
                    "above - do not rewrite from scratch."
                )

            prompt_parts.append(header)
            prompt_parts.append(body)
            prompt_parts.append("\n--- previous Dockerfile (verbatim) ---")
            prompt_parts.append(previous_attempt.get("dockerfile_content", ""))
            if build_error:
                prompt_parts.append("\n--- previous docker build error (tail) ---")
                prompt_parts.append(build_error)
            if gpu_error:
                prompt_parts.append("\n--- previous verify_gpu_runtime error (tail) ---")
                prompt_parts.append(gpu_error)

        if hint:
            prompt_parts.append("\n=== GENERATION HINT (from orchestrator) ===")
            prompt_parts.append(hint)

        output_format_prompt = json.dumps(self.docker_build_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        return "\n".join(prompt_parts)

    def _load_previous_attempt(self, repo_root: Path) -> Optional[Dict[str, str]]:
        """Return the single most recent Dockerfile seen for this repo, paired
        with the error from building it (if any), as
        {'dockerfile_path', 'dockerfile_content', 'build_error'}. Returns
        None only when we have neither a Dockerfile to show nor an error to
        report.

        Semantics of "previous":
          - First dockerfile_generation call: the Dockerfile that shipped
            with the repo. Prefer <repo_root>/Dockerfile; fall back to any
            other Dockerfile discovered in the tree (shortest path wins).
          - Subsequent calls: whatever we wrote to <repo_root>/Dockerfile
            in the prior call (docker_build operates off that path, so it
            is authoritative). We deliberately keep only this one version
            because the stashed build error corresponds to it, and feeding
            multiple earlier Dockerfiles would blur that 1:1 mapping.
        """
        previous_path: Optional[str] = None
        content: str = ""

        root_dockerfile = Path(repo_root) / "Dockerfile"
        if root_dockerfile.is_file():
            try:
                content = root_dockerfile.read_text(encoding="utf-8", errors="ignore")
                previous_path = "Dockerfile"
            except Exception:
                content = ""
        else:
            # First call, original Dockerfile lives at a non-root path.
            try:
                repo_tree = build_tree(repo_root)
                candidates = self.find_dockerfile_files(repo_tree)
            except Exception:
                candidates = []
            # Prefer a file literally named 'Dockerfile', then shortest path.
            candidates.sort(
                key=lambda c: (c.get("name", "").lower() != "dockerfile", len(c.get("path", "")))
            )
            for cand in candidates:
                abs_path = Path(repo_root) / cand.get("path", "")
                if not abs_path.is_file():
                    continue
                try:
                    content = abs_path.read_text(encoding="utf-8", errors="ignore")
                    previous_path = cand.get("path", "")
                    break
                except Exception:
                    continue

        if previous_path is None and not self._last_build_error and not self._last_gpu_runtime_error:
            return None

        # Keep each error tail bounded so one giant log doesn't blow out the
        # context window; docker_build already returns the last ~80 lines and
        # verify_gpu_runtime already trims to ~3000 chars.
        return {
            "dockerfile_path": previous_path or "",
            "dockerfile_content": content,
            "build_error": self._last_build_error[-4000:] if self._last_build_error else "",
            "gpu_runtime_error": self._last_gpu_runtime_error[-4000:] if self._last_gpu_runtime_error else "",
        }

    @staticmethod
    def _fix_multiline_run(dockerfile_text: str) -> str:
        """Fix RUN instructions that span multiple physical lines without
        backslash continuation.

        Docker parses each physical line as a separate instruction, so a RUN
        block like::

            RUN bash -lc 'set -euo pipefail
            if [ ! -f foo.py ]; then
              echo hello
            fi'

        will fail with "unknown instruction: if".  This method detects RUN
        blocks that contain unbalanced quotes (meaning the shell string was
        split across lines) and joins the continuation lines with ' \\\n'
        so Docker sees a single logical instruction.

        The heuristic: after encountering a RUN line whose single-quote or
        double-quote count is odd, keep appending subsequent lines (adding
        a trailing backslash to each) until quotes balance again.
        """
        lines = dockerfile_text.split("\n")
        result: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            # Only process lines that start a RUN instruction
            if stripped.upper().startswith("RUN "):
                # Check if quotes are balanced on this single line
                single_q = stripped.count("'")
                double_q = stripped.count('"')
                if single_q % 2 != 0 or double_q % 2 != 0:
                    # Unbalanced quotes: gather continuation lines
                    merged = [line.rstrip()]
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        next_stripped = next_line.strip()
                        # Stop if we hit another Dockerfile instruction keyword
                        # (and quotes are now balanced) - safety net
                        single_q += next_stripped.count("'")
                        double_q += next_stripped.count('"')
                        merged.append(next_stripped)
                        if single_q % 2 == 0 and double_q % 2 == 0:
                            j += 1
                            break
                        j += 1
                    # Join with backslash continuation
                    fixed = " \\\n    ".join(merged)
                    result.append(fixed)
                    i = j
                    continue
            result.append(line)
            i += 1
        return "\n".join(result)

    def dockerfile_generation(self, repo_root: Path, hint: str = "") -> List[Dict[str, Any]]:
        try:
            self._dockerfile_generation_attempts += 1
            attempt_number = self._dockerfile_generation_attempts

            imports = self.merge_dependencies(repo_root)
            runtime_files = self.detect_runtime_files(repo_root)[0]["value"]

            # Must run BEFORE we open <repo_root>/Dockerfile for write below,
            # otherwise on the first call we'd read our own freshly written
            # file instead of the repo's original Dockerfile.
            previous_attempt = self._load_previous_attempt(repo_root)
            if previous_attempt is not None:
                print(
                    f"[DockerSetUp] dockerfile_generation (attempt #{attempt_number}): "
                    f"feeding previous Dockerfile "
                    f"'{previous_attempt.get('dockerfile_path', '')}' "
                    f"({len(previous_attempt['dockerfile_content'])} chars), "
                    f"{len(previous_attempt.get('build_error', ''))} chars of "
                    f"build error tail, "
                    f"{len(previous_attempt.get('gpu_runtime_error', ''))} chars "
                    f"of verify_gpu_runtime error tail back to the LLM"
                )
            else:
                print(f"[DockerSetUp] dockerfile_generation (attempt #{attempt_number}): first run, no prior attempt context.")

            prompt = self.assemble_docker_build_prompt(
                imports,
                runtime_files,
                self._gpu_profile_for_prompt(),
                previous_attempt,
                attempt_number=attempt_number,
                hint=hint,
            )
            response = json_query(prompt, "dockerfile_generation", self.backend)
            if isinstance(response, str):
                response = json.loads(response)
            dockerfile_path = os.path.join(repo_root, "Dockerfile")
            dockerfile_content = response.get("dockerfile_content", "")
            # Post-generation fix: repair multi-line RUN instructions that
            # the LLM emitted with raw newlines instead of backslash
            # continuation.  Without this, Docker treats continuation lines
            # (e.g. 'if', 'for', 'cat') as unknown Dockerfile instructions.
            dockerfile_content = self._fix_multiline_run(dockerfile_content)
            # Fix double-backslash line continuations (e.g. '\ \' at EOL)
            # that the LLM sometimes emits, which break bash parsing inside
            # RUN instructions.
            dockerfile_content = re.sub(r'\\\s*\\\s*\n', '\\\n', dockerfile_content)
            print("dockerfile_content", dockerfile_content)
            with open(dockerfile_path, "w", encoding="utf-8") as f:
                f.write(dockerfile_content)
            return [{
                "value": dockerfile_path,
                "storage": "permanent",
                "variable_name": "dockerfile_path"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "dockerfile_generation_error"
            }]

    def _resolve_image_name(self, repo_root: Path) -> str:
        configured = (global_config.get("docker_setting") or {}).get("image_name")
        if not configured or str(configured).lower() == "none":
            raw = f"{repo_root.name}_image_{int(time.time())}"
        else:
            raw = str(configured)
        if raw != raw.lower():
            print(f"[DockerSetUp] image_name '{raw}' contains uppercase chars; lowercasing for docker compatibility")
        sanitized = re.sub(r"[^a-z0-9_.-]", "-", raw.lower())
        return sanitized

    def docker_build(self, repo_root: Path, dockerfile_path: Path) -> Dict[str, any]:
        repo_root = Path(repo_root).resolve()
        dockerfile_path = Path(dockerfile_path).resolve()
        image_name = self._resolve_image_name(repo_root)
        print(f"[DockerSetUp] building image '{image_name}' from {dockerfile_path}")

        log_path = repo_root / ".docker_build.log"
        cmd = ["docker", "build", "--network", "host", "-t", image_name, "-f", str(dockerfile_path), str(repo_root)]
        tail = collections.deque(maxlen=80)

        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_file.write(line)
                    tail.append(line.rstrip("\n"))
                    print(line, end="")
                returncode = proc.wait()

            if returncode != 0:
                tail_str = "\n".join(tail)
                raise RuntimeError(
                    f"docker build failed (exit {returncode}). Full log: {log_path}\n"
                    f"--- last {len(tail)} lines ---\n{tail_str}"
                )

            # Success: clear the stash so a later phase retry doesn't carry
            # phantom context from a build that already recovered. Also clear
            # the gpu-runtime error stash - a new image invalidates any prior
            # verify_gpu_runtime failure evidence (which was tied to the old
            # image's torch/CUDA wheel).
            self._last_build_error = ""
            self._last_gpu_runtime_error = ""
            return [{
                "value": image_name,
                "storage": "permanent",
                "variable_name": "image_name",
            }]
        except Exception as e:
            # Stash the error for the next dockerfile_generation call within
            # the same phase execution. _load_previous_attempt() will pick it
            # up together with the current on-disk Dockerfile.
            self._last_build_error = str(e)
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "docker_image_build_error",
            }]

    def docker_image_inspect(self, image_name: str) -> List[Dict[str, Any]]:
        try:
            res = subprocess.run(
                ["docker", "image", "inspect", str(image_name)],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                raise RuntimeError(f"docker image inspect failed: {res.stderr.strip() or res.stdout.strip()}")
            inspect_data = json.loads(res.stdout)
            if not inspect_data:
                raise RuntimeError(f"Image not found: {image_name}")
            return [{
                "value": inspect_data[0],
                "storage": "temporary",
                "variable_name": "docker_image_inspect_result",
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "docker_image_inspect_error",
            }]

    def verify_gpu_runtime(self, image_name: str, dockerfile_path: Path = None) -> List[Dict[str, Any]]:
        """Probe the built image's torch/CUDA stack with a throwaway container.

        Detects torch/CUDA mismatches BEFORE docker_run, so the agent can loop
        back to dockerfile_generation with concrete error evidence rather than
        discovering the failure several phases later in ServiceDelivery as
        an opaque "connection refused" health-check timeout.

        - When ``gpu_profile.gpu_enabled`` is False, this is a no-op (returns a
          'skipped' temporary observation).
        - When the smoke test fails, the returned error embeds the host
          driver_version + cuda_runtime_ceiling so the next dockerfile_generation
          call can pick a compatible torch wheel / base image.
        """
        if not self.gpu_profile.get("gpu_enabled"):
            return [{
                "value": "GPU not enabled in config; GPU runtime smoke test skipped.",
                "storage": "temporary",
                "variable_name": "gpu_runtime_check",
            }]

        # Detect the Python interpreter that the Dockerfile used to install torch.
        # Parsing the Dockerfile directly avoids hardcoding a version list and is
        # more reliable than probing inside the container after the fact.
        # Pattern: any RUN line that calls python3.X explicitly (e.g. pip install
        # under python3.11), or a FROM / ENV that references python3.X.
        python_executable = "python3"
        if dockerfile_path is not None:
            try:
                content = Path(dockerfile_path).read_text(errors="replace")
                # Match the most specific versioned interpreter: python3.XX
                m = re.search(r'\bpython(3\.\d+)\b', content)
                if m:
                    python_executable = f"python{m.group(1)}"
            except Exception:
                pass  # fall back to python3

        gpu_args = self._resolve_gpu_args()
        # Script is passed as a single -c arg, so real newlines + indent are
        # fine. The early cpu_only_wheel exit avoids flagging an intentional
        # CPU fallback (rule 24 in dockerfile_prompt.json) as a GPU mismatch.
        probe_script = (
            "import torch, sys\n"
            "print('torch_version=' + torch.__version__)\n"
            "print('torch_cuda=' + str(torch.version.cuda))\n"
            "if torch.version.cuda is None:\n"
            "    print('cpu_only_wheel')\n"
            "    sys.exit(0)\n"
            "assert torch.cuda.is_available(), 'torch.cuda.is_available() returned False'\n"
            "x = torch.zeros(1).cuda()\n"
            "y = x + 1\n"
            "print('cuda_smoke_ok')\n"
        )
        cmd: List[str] = ["docker", "run", "--rm"] + gpu_args + [
            str(image_name),
            python_executable, "-c", probe_script,
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            msg = (
                "GPU runtime smoke test timed out after 180s. "
                f"host gpu_profile: driver_version={self.gpu_profile.get('driver_version')}, "
                f"cuda_runtime_ceiling={self.gpu_profile.get('cuda_runtime_ceiling')}. "
                "Re-run dockerfile_generation with this evidence and rebuild."
            )
            self._last_gpu_runtime_error = msg
            return [{
                "value": msg,
                "storage": "error",
                "variable_name": "gpu_runtime_mismatch",
            }]
        except Exception as e:
            msg = f"Failed to launch GPU smoke test: {e}"
            self._last_gpu_runtime_error = msg
            return [{
                "value": msg,
                "storage": "error",
                "variable_name": "gpu_runtime_mismatch",
            }]

        # Deliberate CPU fallback (rule 24): the image installs a CPU-only
        # torch wheel, so torch.version.cuda is None and the GPU smoke test
        # is not meaningful. Clear the gpu error stash and report as a
        # non-error skip so the agent does not loop back into
        # dockerfile_generation.
        if res.returncode == 0 and "cpu_only_wheel" in (res.stdout or ""):
            self._last_gpu_runtime_error = ""
            return [{
                "value": (
                    "Image ships a CPU-only torch wheel (torch.version.cuda is None). "
                    "GPU runtime smoke test intentionally skipped - this is an "
                    "accepted CPU fallback, not a mismatch. Service will run on CPU."
                ),
                "storage": "temporary",
                "variable_name": "gpu_runtime_check",
            }]

        if res.returncode != 0 or "cuda_smoke_ok" not in (res.stdout or ""):
            combined = (res.stdout or "") + (res.stderr or "")
            tail = combined[-3000:]

            # Distinguish docker launch/daemon failure from in-container torch
            # failure. The probe never executed when docker rejected the run
            # itself (bad --gpus args, missing runtime, image pull error, etc.)
            # - misclassifying that as a torch/CUDA mismatch would send
            # dockerfile_generation into a loop of useless Dockerfile
            # rewrites. Surface it as a separate error variable, and DO NOT
            # stash it into _last_gpu_runtime_error (which is the feedback
            # channel specifically for torch/CUDA image problems).
            launch_failure_markers = (
                "docker: Error response from daemon",
                "Error response from daemon:",
                "cannot set both Count and DeviceIDs",
                "could not select device driver",
                "nvidia-container-cli",
                "Unable to find image",
                "pull access denied",
                "manifest unknown",
                "OCI runtime create failed",
            )
            is_launch_failure = any(m in combined for m in launch_failure_markers) and (
                "torch_version=" not in combined  # probe never ran
            )

            if is_launch_failure:
                launch_msg = (
                    "GPU runtime smoke test could NOT launch the container - "
                    "this is a docker/runtime issue, NOT a torch/CUDA image "
                    "problem. Do not rebuild the Dockerfile in response. "
                    "Check --gpus args, NVIDIA container toolkit, and image "
                    f"availability.\n--- docker output (tail) ---\n{tail}"
                )
                # Keep whatever torch/cuda mismatch we stashed before (if any)
                # untouched; this launch failure is orthogonal and must not
                # overwrite it or invent a phantom mismatch.
                return [{
                    "value": launch_msg,
                    "storage": "error",
                    "variable_name": "gpu_runtime_launch_error",
                }]

            # torch not installed at all — completely different from a CUDA driver
            # mismatch. The probe died on the very first `import torch` line, which
            # means the Dockerfile never installed torch (or pip install failed
            # silently). The fix is to correct the pip install step, NOT to change
            # the base image or the CUDA version.
            if "No module named 'torch'" in combined:
                no_torch_msg = (
                    "torch is not installed in the image "
                    "(ModuleNotFoundError: No module named 'torch'). "
                    "This is NOT a CUDA driver mismatch — torch was never installed. "
                    "Re-run dockerfile_generation to add a correct 'pip install torch' "
                    "step, then rebuild and re-run verify_gpu_runtime. "
                    "Do NOT change the base image CUDA version in response to this error.\n"
                    f"--- smoke test output (tail) ---\n{tail}"
                )
                self._last_gpu_runtime_error = no_torch_msg
                return [{
                    "value": no_torch_msg,
                    "storage": "error",
                    "variable_name": "gpu_runtime_mismatch",
                }]

            msg = (
                "Image's torch/CUDA stack is incompatible with the host driver. "
                "Do NOT proceed to docker_run. Re-run dockerfile_generation using the "
                "host driver_version + cuda_runtime_ceiling below to pick a compatible "
                "torch wheel / base image, then rebuild and re-run verify_gpu_runtime.\n"
                f"host gpu_profile: driver_version={self.gpu_profile.get('driver_version')}, "
                f"cuda_runtime_ceiling={self.gpu_profile.get('cuda_runtime_ceiling')}, "
                f"selected_gpus={self.gpu_profile.get('selected_gpus')}\n"
                f"--- smoke test output (tail) ---\n{tail}"
            )
            self._last_gpu_runtime_error = msg
            return [{
                "value": msg,
                "storage": "error",
                "variable_name": "gpu_runtime_mismatch",
            }]

        # Success: clear the stash so a subsequent dockerfile_generation
        # (should one still be triggered) does not carry stale evidence.
        self._last_gpu_runtime_error = ""
        return [{
            "value": (res.stdout or "").strip(),
            "storage": "temporary",
            "variable_name": "gpu_runtime_check",
        }]

    def _resolve_container_name(self, image_name: str) -> str:
        configured = (global_config.get("docker_setting") or {}).get("container_name")
        if configured is None or str(configured).strip() == "" or str(configured).lower() == "none":
            raw = f"{image_name}_container_{int(time.time())}"
        else:
            raw = str(configured)
        return re.sub(r"[^a-z0-9_.-]", "-", raw.lower())

    def _resolve_port_args(self) -> List[str]:
        """Build a list of `-p host:container` flags from yaml config.

        Supports `port_mapping` as None, a single string, or a list.
        """
        port_mapping = (global_config.get("docker_setting") or {}).get("port_mapping")
        if port_mapping is None or port_mapping == "":
            return ["-p", "8000:8000"]
        if isinstance(port_mapping, list):
            args: List[str] = []
            for p in port_mapping:
                p_str = str(p).strip()
                if p_str:
                    args.extend(["-p", p_str])
            return args or ["-p", "8000:8000"]
        return ["-p", str(port_mapping).strip()]

    def _resolve_gpu_args(self) -> List[str]:
        """Build `--gpus ...` args from the precomputed gpu_profile (#10).

        Single source of truth: self.gpu_profile (already parsed in __init__).

        Multi-GPU quoting: Docker CSV-parses the value of `--gpus`, so a bare
        `device=0,1` gets split into [`device=0`, `1`] and the trailing `1`
        is interpreted as `count=1` - Docker then refuses the request with
        "cannot set both Count and DeviceIDs on device request". Wrapping the
        value in literal double quotes (`"device=0,1"`) tells Docker's CSV
        parser to keep the comma inside a single field.
        """
        if not self.gpu_profile.get("gpu_enabled"):
            return []
        indices = self.gpu_profile.get("selected_gpu_indices") or []
        if not indices:
            return ["--gpus", "all"]
        if len(indices) == 1:
            return ["--gpus", f"device={indices[0]}"]
        joined = ",".join(str(i) for i in indices)
        return ["--gpus", f'"device={joined}"']

    def _update_config_port_mapping(self, new_mapping: str) -> None:
        global_config.setdefault("docker_setting", {})["port_mapping"] = new_mapping
        config_path = os.path.join(project_path, "config/global.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(global_config, f, default_flow_style=False, allow_unicode=True)
        print(f"[DockerSetUp] updated config/global.yaml port_mapping to {new_mapping}")

    def _is_port_in_use(self, port: int) -> bool:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return False
            except OSError:
                return True

    def _free_host_port(self) -> int:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            return s.getsockname()[1]

    def _parse_port_arg_pair(self, port_args: List[str]) -> Optional[tuple]:
        """Extract (host_port, container_port) from the first `-p host:ctr`
        pair in port_args. Returns None when the mapping is anything exotic
        (protocol suffix, ranges, IP-bound) that the fallback logic should
        not touch.
        """
        for i, tok in enumerate(port_args):
            if tok == "-p" and i + 1 < len(port_args):
                spec = port_args[i + 1].strip()
                parts = spec.split(":")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    return int(parts[0]), int(parts[1])
                return None
        return None

    def _cleanup_containers_on_port(self, host_port: int) -> None:
        """Force-remove any existing containers that publish the given host
        port. Catches the common 'previous run crashed and left a container
        around' case that docker_run's same-name cleanup misses when the
        image_name-based container name has a fresh timestamp.
        """
        ps = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"publish={host_port}", "-q"],
            capture_output=True,
            text=True,
        )
        ids = [cid.strip() for cid in (ps.stdout or "").splitlines() if cid.strip()]
        if not ids:
            return
        print(f"[DockerSetUp] removing {len(ids)} container(s) publishing host port {host_port}: {ids}")
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, text=True)

    def docker_run(self, repo_root: Path, image_name: str) -> Dict[str, any]:
        repo_root = Path(repo_root).resolve()
        work_dir = (global_config.get("docker_setting") or {}).get("work_dir") or "/workspace"

        container_name = self._resolve_container_name(image_name)
        port_args = self._resolve_port_args()
        gpu_args = self._resolve_gpu_args()

        port_pair = self._parse_port_arg_pair(port_args)
        if port_pair is not None:
            host_port, container_port = port_pair
            if self._is_port_in_use(host_port):
                new_port = self._free_host_port()
                print(
                    f"[DockerSetUp] host port {host_port} is in use; "
                    f"switching to free port {new_port} -> container port {container_port}"
                )
                new_mapping = f"{new_port}:{container_port}"
                port_args = ["-p", new_mapping]
                port_pair = (new_port, container_port)
                self._update_config_port_mapping(new_mapping)

        def _build_cmd(current_port_args: List[str]) -> List[str]:
            c = ["docker", "run", "-d", "--init", "--name", container_name]
            c.extend(gpu_args)
            c.extend(["-v", f"{repo_root}:{work_dir}"])
            c.extend(current_port_args)
            # Keep the container alive as a long-lived exec target. Without an
            # explicit CMD, the image's default (typically `bash` on nvidia
            # CUDA bases) exits immediately under `-d` (no TTY/stdin), and
            # --restart=unless-stopped pulls it back into an infinite
            # banner-print + exit loop. `sleep infinity` is a PID-1 that
            # simply blocks forever, so `docker exec` always has a live
            # namespace to attach to.
            c.extend(["--restart", "unless-stopped", "--entrypoint", "sleep", image_name, "infinity"])
            return c

        cmd: List[str] = _build_cmd(port_args)
        effective_port_mapping = port_args  # tracked so we can report actual bindings

        try:
            run_res = subprocess.run(cmd, capture_output=True, text=True)

            # Port-conflict auto-retry: if the host port is still taken (e.g.
            # by a non-docker process or a container we can't see), pick a
            # fresh free port and retry exactly once. The downstream reader
            # (ServiceDelivery._get_runtime_ports) queries `docker port` on
            # the container, so it picks up whatever we actually bound.
            if run_res.returncode != 0 and port_pair is not None:
                combined = (run_res.stderr or "") + (run_res.stdout or "")
                if "port is already allocated" in combined or "Bind for " in combined:
                    new_host_port = self._free_host_port()
                    host_port_original, container_port = port_pair
                    print(
                        f"[DockerSetUp] host port {host_port_original} still in use "
                        f"after cleanup; retrying with free port {new_host_port} -> "
                        f"container port {container_port}"
                    )
                    retry_mapping = f"{new_host_port}:{container_port}"
                    new_port_args = ["-p", retry_mapping]
                    self._update_config_port_mapping(retry_mapping)
                    # Same-name container from our first failed attempt may
                    # linger in 'created' state; remove before retry.
                    subprocess.run(
                        ["docker", "rm", "-f", container_name],
                        capture_output=True, text=True,
                    )
                    cmd = _build_cmd(new_port_args)
                    effective_port_mapping = new_port_args
                    run_res = subprocess.run(cmd, capture_output=True, text=True)

            if run_res.returncode != 0:
                raise RuntimeError(
                    f"docker run failed (exit {run_res.returncode}): "
                    f"{run_res.stderr.strip() or run_res.stdout.strip()}"
                )
            container_id = run_res.stdout.strip()

            # #6: poll for liveness up to 10s, capture logs on early exit.
            # Goes through inspect_running so this loop stays aligned with the
            # other two callers (APIAdaptation/ServiceDelivery.check_container_status).
            # Critically, inspect_running rejects `restarting` - if the container
            # immediately enters a restart loop after launch, we catch it here
            # instead of returning success and letting downstream docker exec
            # calls hit the daemon's "Container is restarting" error.
            exited = False
            for _ in range(10):
                time.sleep(1)
                if not inspect_running(container_id):
                    exited = True
                    break

            if exited:
                logs_res = subprocess.run(
                    ["docker", "logs", "--tail", "200", container_id],
                    capture_output=True,
                    text=True,
                )
                logs = (logs_res.stdout or "") + (logs_res.stderr or "")
                logs_tail = logs[-3000:] if logs else "(no logs captured)"
                raise RuntimeError(
                    f"Container '{container_name}' exited shortly after start.\n"
                    f"--- docker logs (tail) ---\n{logs_tail}"
                )

            # Record the actual -p mapping we ended up using. ServiceDelivery
            # prefers `docker port <id>` over this, but this is a cheap backup
            # for environments where the inspect call is unreliable.
            actual_mapping = ""
            for i, tok in enumerate(effective_port_mapping):
                if tok == "-p" and i + 1 < len(effective_port_mapping):
                    actual_mapping = effective_port_mapping[i + 1]
                    break

            return [{
                    "value": container_name,
                    "storage": "permanent",
                    "variable_name": "container_name"
                },
                {
                    "value": container_id,
                    "storage": "permanent",
                    "variable_name": "container_id"
                },
                {
                    "value": actual_mapping,
                    "storage": "permanent",
                    "variable_name": "runtime_port_mapping"
                },
                {
                    "value": "successfully run the Docker container",
                    "storage": "temporary",
                    "variable_name": "run_docker_container"
                }]

        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "docker_container_running_error"
            }]

    def read_file(self, path: str) -> List[Dict[str, Any]]:
        """Read file content. For large files, returns only the last 200 lines."""
        try:
            p = Path(path)
            if not p.is_file():
                return [{
                    "value": f"File not found: {path}",
                    "storage": "error",
                    "variable_name": "read_file_error"
                }]
            content = p.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            if len(lines) > 200:
                content = "\n".join(lines[-200:])
                content = f"[... truncated, showing last 200 of {len(lines)} lines ...]\n" + content
            return [{
                "value": content,
                "storage": "temporary",
                "variable_name": "file_content"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "read_file_error"
            }]
