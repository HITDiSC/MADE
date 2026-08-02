import os
import subprocess
import re
import threading
from typing import List, Dict, Any, Tuple, Optional
import shlex
from pathlib import Path
from typing import Union
import ast
import traceback

class VariableStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._store: List[Dict[str, Any]] = []

    def get(self, name: str) -> Tuple[bool, Any]:
        with self._lock:
            for d in reversed(self._store):
                if name in d:
                    return True, d[name]
            return False, {"missing": name}

    def get_many(self, names: List[str]) -> Tuple[bool, Dict[str, Any]]:
        result = {}
        with self._lock:
            for name in names:
                found = False
                for d in reversed(self._store):
                    if name in d:
                        result[name] = d[name]
                        found = True
                        break
                if not found:
                    return False, {"missing": name}
        return True, result

    def get_all(self) -> Dict[str, Any]:
        result = {}
        with self._lock:
            for d in self._store:
                result.update(d)
        return result

    def write(self, name: str, value: Any) -> None:
        with self._lock:
            self._store.append({name: value})

def em_args_to_dict(em_arguments):
    """
    Normalize em_arguments into a dict.

    Supported:
    1) {"pdf_path": "..."}
    2) [{"pdf_path": "..."}, {"x": 1}]
    3) [{"key": "pdf_path", "value": "..."}, ...]
    """
    if em_arguments is None:
        return {}

    # case 1: already dict
    if isinstance(em_arguments, dict):
        return em_arguments

    # case 2/3: list
    if isinstance(em_arguments, list):
        out = {}
        for item in em_arguments:
            if not isinstance(item, dict):
                continue
            # case 3: {"key": ..., "value": ...}
            if "key" in item and "value" in item and len(item) == 2:
                k = item.get("key")
                if isinstance(k, str):
                    out[k] = item.get("value")
            else:
                # case 2: normal dict merge
                out.update(item)
        return out

    # fallback
    return {}

def get_arguments(primary_store, secondary_store, em_arguments, names):
    result = {}
    em_dict = em_args_to_dict(em_arguments)

    for name in names:
        found, value = primary_store.get(name)
        if found:
            result[name] = value
            continue

        found, value = secondary_store.get(name)
        if found:
            result[name] = value
            continue

        # fallback to em_arguments
        if name in em_dict:
            result[name] = em_dict[name]
            continue

        return False, {"missing": name}

    return True, result

def build_tree(path, max_depth=3, _current_depth=0):
    if not os.path.exists(path):
        return {"name": os.path.basename(path), "type": "error", "message": f"path does not exist: {path}"}
    node = {
        "name": os.path.basename(path),
        "size": os.path.getsize(path),
        "type": "dir" if os.path.isdir(path) else "file"
    }
    if os.path.isdir(path):
        if _current_depth >= max_depth:
            try:
                children_count = len(os.listdir(path))
            except Exception:
                children_count = "?"
            node["children"] = f"... ({children_count} items, use a deeper path to explore)"
        else:
            node["children"] = [
                build_tree(os.path.join(path, p), max_depth=max_depth, _current_depth=_current_depth + 1)
                for p in os.listdir(path)
            ]
    return node

def linux_command(commands: str):
    try:
        result = subprocess.run(
            commands,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        raise RuntimeError(f"Command failed: {e}\n{error_trace}")

def linux_command_in_docker(command: str, container_id: str, workdir: str | None = None):
    """
    Execute a shell command inside a Docker container safely.
    """
    try:
        safe_cmd = shlex.quote(command)

        if workdir:
            safe_workdir = shlex.quote(workdir)
            full_cmd = f"docker exec -w {safe_workdir} {container_id} bash -c {safe_cmd}"
        else:
            full_cmd = f"docker exec {container_id} bash -c {safe_cmd}"

        return linux_command(full_cmd)

    except Exception as e:
        error_trace = traceback.format_exc()
        raise RuntimeError(f"Command failed: {e}\n{error_trace}")


def inspect_running(container_id: str) -> bool:
    """Return True iff the docker container is currently in 'running' state.

    Single source of truth used by DockerSetUp.docker_run polling,
    ServiceDelivery.check_container_status, and APIAdaptation.check_container_status.
    Never raises - a missing/exited/unknown container is reported as False so callers
    can branch on a single boolean.
    """
    # Check `.State.Status` rather than `.State.Running`: Docker reports
    # Running=true even while a container is in the `restarting` state, during
    # which `docker exec` is rejected by the daemon. Callers need to know the
    # container is actually usable, not merely "not stopped".
    try:
        res = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", str(container_id)],
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    if res.returncode != 0:
        return False
    return res.stdout.strip() == "running"

def locate_local_path(file_paths: List[str], query_path: str) -> Tuple[bool, Optional[str]]:
    """
    Locate a local path by suffix matching only.

    Returns:
        (found: bool, path: Optional[str])
    """

    def normalize(p: str) -> str:
        p = (p or "").strip().replace("\\", "/").lower()
        p = re.sub(r"/+", "/", p)
        return p.strip("/")

    q = normalize(query_path)
    if not q:
        return False, None

    for p in file_paths:
        p_norm = normalize(p)
        # core rule: suffix match
        if p_norm.endswith(q):
            return True, p

    return False, None

def locate_path(repo_root: Path, query: Union[str, Path]) -> Tuple[bool, Optional[Union[str, List[str]]], Optional[str]]:
    """
    Supports:
    1. full relative-path match (preferred)
    2. name match (fallback)
    3. both files and folders
    4. the input query may be:
       - a full absolute path (including repo_root)
       - an absolute / pseudo-absolute path containing the repo name
       - a path relative to repo_root
    5. returns the normalized absolute path

    For example, when:
        repo_root = /home/user/repo

    the following queries are all normalized to:
        subdir/configs/example.yaml

        /home/user/repo/subdir/configs/example.yaml
        /repo/subdir/configs/example.yaml
        /subdir/configs/example.yaml

    Returns:
        (found, result, node_type)

        - found=True:
            result -> absolute path str
            node_type -> "file" or "dir"

        - found=False and multiple candidates:
            result -> list of absolute paths List[str]
            node_type -> None

        - found=False and no match:
            result -> None
            node_type -> None
    """
    repo_root = repo_root.resolve()
    repo_root_str = str(repo_root).replace("\\", "/").rstrip("/")
    repo_name = repo_root.name

    if isinstance(query, Path):
        query = str(query)

    query = query.replace("\\", "/").strip()

    def normalize_query_to_repo_relative(q: str) -> str:
        """
        Normalize various input path forms into a path relative to repo_root.
        """
        q = q.replace("\\", "/").strip()

        if not q:
            return ""

        # case 1: full absolute path that includes repo_root
        # e.g. /home/user/repo/subdir/configs/example.yaml
        if q.startswith(repo_root_str + "/"):
            return q[len(repo_root_str) + 1:].strip("/")

        if q == repo_root_str:
            return ""

        # case 2: contains the repo name but not the full repo_root prefix
        # e.g. /repo/subdir/configs/example.yaml
        marker = f"/{repo_name}/"
        if marker in q:
            return q.split(marker, 1)[1].strip("/")

        # case 3: exactly /repo
        if q.rstrip("/") == f"/{repo_name}":
            return ""

        # case 4: pseudo-absolute path relative to repo_root
        # e.g. /subdir/configs/example.yaml
        if q.startswith("/"):
            return q.strip("/")

        # case 5: already a relative path
        return q.strip("/")

    normalized_query = normalize_query_to_repo_relative(query)
    query_name = Path(normalized_query).name if normalized_query else repo_name

    candidates = []

    for p in repo_root.rglob("*"):
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        abs_path = str(p.resolve()).replace("\\", "/")
        node_type = "dir" if p.is_dir() else "file"

        # 1. prefer a full relative-path match
        if rel == normalized_query:
            return True, abs_path, node_type

        # 2. fallback: match by name
        if p.name == query_name:
            candidates.append((abs_path, node_type))

    # if the query points at repo_root itself
    if normalized_query == "":
        return True, repo_root_str, "dir"

    if len(candidates) == 1:
        return True, candidates[0][0], candidates[0][1]
    elif len(candidates) > 1:
        return False, [c[0] for c in candidates], None
    else:
        return False, None, None

def path_to_import(repo_root: Path, predict_dir: Path,predict_file = "predict.py") -> str | None:
    """
    Convert file path to Python import path.
    Returns None if any path component is not a valid Python identifier
    (e.g. contains '-', '.', or starts with a digit).

    Example:
        /root/project/src/api + predict.py
        -> src.api.predict
    """
    try:
        repo_root = Path(repo_root)
        predict_dir = Path(predict_dir)
        relative = predict_dir.relative_to(repo_root)
    except ValueError:
        raise ValueError("predict_dir must be under repo_root")

    file_name = predict_file.replace(".py", "")
    parts = list(relative.parts) + [file_name]

    for part in parts:
        if not part.isidentifier():
            return None

    module_path = ".".join(relative.parts)

    if module_path:
        return f"{module_path}.{file_name}"
    else:
        return file_name

def replace_line(file_path: str, target_prefix: str, new_line: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip().startswith(target_prefix):
            new_lines.append(new_line + "\n")
        else:
            new_lines.append(line)

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def get_local_import_lines(file_path: str, project_root: str) -> List[str]:
    file_path = Path(file_path).resolve()
    project_root = Path(project_root).resolve()

    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    lines = source.splitlines()

    local_imports = set()

    def exists_module(base: Path, parts: List[str]) -> bool:
        """Return whether the module exists."""
        py_file = base.joinpath(*parts).with_suffix(".py")
        init_file = base.joinpath(*parts, "__init__.py")
        return py_file.exists() or init_file.exists()

    for node in ast.walk(tree):

        # ---------- import xxx ----------
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")

                if exists_module(project_root, parts):
                    local_imports.add(lines[node.lineno - 1])

        # ---------- from xxx import ----------
        elif isinstance(node, ast.ImportFrom):

            # ===== relative import =====
            if node.level > 0:
                base = file_path.parent

                # walk up `level` parent packages
                for _ in range(node.level - 1):
                    base = base.parent

                if node.module:
                    parts = node.module.split(".")
                    if exists_module(base, parts):
                        local_imports.add(lines[node.lineno - 1])
                else:
                    # from . import xxx
                    local_imports.add(lines[node.lineno - 1])

            # ===== absolute import =====
            else:
                if node.module:
                    parts = node.module.split(".")
                    if exists_module(project_root, parts):
                        local_imports.add(lines[node.lineno - 1])

    return list(local_imports)


def _resolve_module_to_file(module_parts: List[str], search_root: Path) -> Optional[Path]:
    """Resolve dotted module parts to a .py file under *search_root*.

    Tries ``search_root / a / b / c.py`` first, then
    ``search_root / a / b / c / __init__.py``.  Returns the resolved
    ``Path`` or ``None`` if neither exists.
    """
    py_file = search_root.joinpath(*module_parts).with_suffix(".py")
    if py_file.exists():
        return py_file.resolve()
    init_file = search_root.joinpath(*module_parts, "__init__.py")
    if init_file.exists():
        return init_file.resolve()
    return None


def resolve_import_closure(seed_files: List[Union[str, Path]],
                           project_root: Union[str, Path]) -> List[str]:
    """Recursively collect every local .py file reachable via imports.

    Starting from *seed_files*, parse each file's AST to extract
    ``import`` / ``from ... import`` statements that resolve to files
    under *project_root*, then recurse until the closure is stable.

    Returns a deduplicated list of absolute path strings, sorted for
    determinism.
    """
    project_root = Path(project_root).resolve()
    seen: set[Path] = set()
    queue: list[Path] = []

    for f in seed_files:
        p = Path(f).resolve()
        if p.exists() and p.suffix == ".py":
            queue.append(p)

    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)

        try:
            source = current.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            resolved: Optional[Path] = None

            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    resolved = _resolve_module_to_file(parts, project_root)
                    if resolved and resolved not in seen:
                        queue.append(resolved)

            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    # Relative import
                    base = current.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    if node.module:
                        parts = node.module.split(".")
                        resolved = _resolve_module_to_file(parts, base)
                    # ``from . import X`` — check if X is a submodule
                    if resolved is None and node.names:
                        for alias in node.names:
                            sub = _resolve_module_to_file([alias.name], base)
                            if sub and sub not in seen:
                                queue.append(sub)
                else:
                    if node.module:
                        parts = node.module.split(".")
                        resolved = _resolve_module_to_file(parts, project_root)

                if resolved and resolved not in seen:
                    queue.append(resolved)

    return sorted(str(p) for p in seen)