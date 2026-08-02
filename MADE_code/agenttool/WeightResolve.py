import json
import os
import re
from abc import ABC, abstractmethod
import yaml
from pathlib import Path
from typing import Union, List, Dict, Callable, Optional, Any
import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Union, List
from huggingface_hub import list_repo_tree, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from backend.query import query, json_query
import gdown
import time
from agenttool.base_phase import BasePhase
from agenttool.tool import build_tree, locate_local_path, linux_command, locate_path
from agenttool.gpu_profile import resolve_gpu_profile

file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)


try:
    with open(os.path.join(project_path, "config/global.yaml"), "r") as f:
        global_config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found.")
except yaml.YAMLError as exc:
    raise yaml.YAMLError(f"Error in configuration file: {exc}")

GOOGLE_DRIVE_API_KEY = global_config.get("backend").get("google_drive_api_key")
KAGGLE_USERNAME = global_config.get("backend").get("kaggle_username")
KAGGLE_API_KEY = global_config.get("backend").get("kaggle_api_key")

@dataclass
class DriveFile:
    path: str
    size: int
    id: str
    mime_type: str


@dataclass
class DriveFolder:
    path: str
    id: str


WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".pb", ".h5", ".hdf5", ".tflite", ".onnx"}
# Non-weight files that are commonly required at inference time
# (tokenizer, vocabulary, sentencepiece model, config, etc.)
AUXILIARY_SUFFIXES = {".json", ".txt", ".model", ".spm", ".tiktoken", ".vocab"}
# Top-level repo metadata files that are never needed at inference time
AUXILIARY_EXCLUDE_NAMES = {"README.md", ".gitattributes", "LICENSE", "LICENSE.txt"}
WEIGHTS_SUBDIR = "weights_related"
PENDING_REGISTRY_RELPATH = Path(".autodeploy") / "pending_weight_candidates.json"


def _registry_path(repo_root: Path) -> Path:
    return Path(repo_root) / PENDING_REGISTRY_RELPATH


def _load_registry(repo_root: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    p = _registry_path(repo_root)
    if not p.exists():
        return {"huggingface": {}, "google_drive": {}, "kaggle": {}, "zenodo": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data.setdefault("huggingface", {})
    data.setdefault("google_drive", {})
    data.setdefault("kaggle", {})
    data.setdefault("zenodo", {})
    return data


def _save_registry(repo_root: Path, data: Dict[str, Any]) -> Path:
    p = _registry_path(repo_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return p


def _register_hf_candidates(repo_root: Path, hf_repo_id: str, candidates: List[Dict[str, Any]]) -> Path:
    data = _load_registry(repo_root)
    bucket = data["huggingface"].setdefault(hf_repo_id, [])
    existing_paths = {c.get("path") for c in bucket}
    for c in candidates:
        if c.get("path") not in existing_paths:
            bucket.append(c)
            existing_paths.add(c.get("path"))
    return _save_registry(repo_root, data)


def _register_gd_candidates(repo_root: Path, gd_url: str, candidates: List[Dict[str, Any]]) -> Path:
    data = _load_registry(repo_root)
    bucket = data["google_drive"].setdefault(gd_url, [])
    existing_ids = {c.get("id") for c in bucket}
    for c in candidates:
        if c.get("id") not in existing_ids:
            bucket.append(c)
            existing_ids.add(c.get("id"))
    return _save_registry(repo_root, data)


def _remove_hf_candidate(repo_root: Path, hf_repo_id: str, file_path: str) -> None:
    data = _load_registry(repo_root)
    bucket = data["huggingface"].get(hf_repo_id)
    if not bucket:
        return
    data["huggingface"][hf_repo_id] = [c for c in bucket if c.get("path") != file_path]
    if not data["huggingface"][hf_repo_id]:
        del data["huggingface"][hf_repo_id]
    _save_registry(repo_root, data)


def _remove_gd_candidate(repo_root: Path, file_id: str) -> None:
    data = _load_registry(repo_root)
    changed = False
    for url in list(data["google_drive"].keys()):
        filtered = [c for c in data["google_drive"][url] if c.get("id") != file_id]
        if len(filtered) != len(data["google_drive"][url]):
            changed = True
        if filtered:
            data["google_drive"][url] = filtered
        else:
            del data["google_drive"][url]
    if changed:
        _save_registry(repo_root, data)


def _register_zenodo_candidates(repo_root: Path, record_id: str, candidates: List[Dict[str, Any]]) -> Path:
    data = _load_registry(repo_root)
    bucket = data["zenodo"].setdefault(record_id, [])
    existing_keys = {c.get("key") for c in bucket}
    for c in candidates:
        if c.get("key") not in existing_keys:
            bucket.append(c)
            existing_keys.add(c.get("key"))
    return _save_registry(repo_root, data)


def _remove_zenodo_candidate(repo_root: Path, record_id: str, file_key: str) -> None:
    data = _load_registry(repo_root)
    bucket = data["zenodo"].get(record_id)
    if not bucket:
        return
    data["zenodo"][record_id] = [c for c in bucket if c.get("key") != file_key]
    if not data["zenodo"][record_id]:
        del data["zenodo"][record_id]
    _save_registry(repo_root, data)


def _download_zenodo_file(repo_root: Path, download_url: str, filename: str) -> Path:
    weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / filename
    resp = requests.get(download_url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(target, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return target


def _register_kaggle_candidates(repo_root: Path, kaggle_handle: str, candidates: List[Dict[str, Any]]) -> Path:
    data = _load_registry(repo_root)
    bucket = data["kaggle"].setdefault(kaggle_handle, [])
    existing_paths = {c.get("path") for c in bucket}
    for c in candidates:
        if c.get("path") not in existing_paths:
            bucket.append(c)
            existing_paths.add(c.get("path"))
    return _save_registry(repo_root, data)


def _remove_kaggle_candidate(repo_root: Path, kaggle_handle: str, file_path: str) -> None:
    data = _load_registry(repo_root)
    bucket = data["kaggle"].get(kaggle_handle)
    if not bucket:
        return
    data["kaggle"][kaggle_handle] = [c for c in bucket if c.get("path") != file_path]
    if not data["kaggle"][kaggle_handle]:
        del data["kaggle"][kaggle_handle]
    _save_registry(repo_root, data)


def _setup_kaggle_credentials():
    if KAGGLE_USERNAME and KAGGLE_API_KEY:
        os.environ.setdefault("KAGGLE_USERNAME", KAGGLE_USERNAME)
        os.environ.setdefault("KAGGLE_KEY", KAGGLE_API_KEY)


def _download_kaggle_model(repo_root: Path, kaggle_handle: str) -> Path:
    _setup_kaggle_credentials()
    import kagglehub
    cache_path = Path(kagglehub.model_download(kaggle_handle))
    weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    target_dir = weights_dir / kaggle_handle.replace("/", "_")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(cache_path, target_dir)
    return target_dir


def _download_google_drive_file(repo_root: Path, file_id: str) -> Path:
    weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    downloaded = gdown.download(url, quiet=False)
    if not downloaded:
        raise RuntimeError("Google Drive download returned no file path.")
    target = weights_dir / Path(downloaded).name
    if Path(downloaded).resolve() != target.resolve():
        shutil.move(downloaded, target)
    return target


def _download_huggingface_file(repo_root: Path, hf_repo_id: str, file_path: str) -> Path:
    weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(
        repo_id=hf_repo_id,
        filename=file_path,
        local_dir=str(weights_dir),
    )
    return Path(p)


class WeightResolve(BasePhase):
    name: str = "WeightResolve"
    description: str = "Resolve the Weights for the model."
    goal: str = "get the Weights"
    tools_schemas: List[Dict[str, any]] = [
        {"name": "get_file_tree", "description": "Get the tree of a local folder when you need to inspect where weights or weight references may exist.", "args": {"path": Path}},
        {"name": "read_readme_content", "description": "Read the cleaned README content so you can extract weight download clues such as checkpoint links, Hugging Face repo ids, Google Drive folders, and expected weight file names.", "args": {"cleaned_readmes_path": Path}},
        {"name": "read_text_resource", "description": "Read a local text file such as README, config, or script content when you need to extract weight download clues.", "args": {"repo_root": Path, "path": Path}},
        {"name": "inspect_remote_weight_source", "description": "Inspect a remote weight source URL and normalize it into a Google Drive, Hugging Face, Kaggle, or Zenodo source with downloadable candidates. Also persists the candidate list to <repo_root>/.autodeploy/pending_weight_candidates.json so later phases (e.g. ServiceDelivery) can pull any still-undownloaded files.", "args": {"repo_root": Path, "source": str}},
        {"name": "find_local_weights", "description": "Look for local model weight files inside the repository. On success, writes weights_path to the variable store; the LLM must still call verify_weights_path before the phase can terminate. NOTE: finding local weights does NOT populate the pending registry. If the model has a known Hugging Face repo or Google Drive source, still call get_huggingface_candidates or get_google_drive_candidates after this tool to register auxiliary files (tokenizer, config, vocab, etc.) so ServiceDelivery can download them if needed at inference time.", "args": {"repo_root": Path}},
        {"name": "get_google_drive_candidates", "description": "List downloadable weight candidates from a Google Drive folder URL or folder id. Persists the candidate list to <repo_root>/.autodeploy/pending_weight_candidates.json.", "args": {"repo_root": Path, "gd_url": str}},
        {"name": "get_huggingface_candidates", "description": "List downloadable weight candidates from a Hugging Face repo id. Persists the candidate list to <repo_root>/.autodeploy/pending_weight_candidates.json.", "args": {"repo_root": Path, "hf_repo_id": str}},
        {"name": "google_drive_weights_download", "description": "Download a selected weight file from Google Drive using its file id.", "args": {"repo_root": Path, "file_id": str}},
        {"name": "huggingface_weights_download", "description": "Download a selected weight file from Hugging Face using repo id and file path.", "args": {"repo_root": Path, "hf_repo_id": str, "file_path": str}},
        {"name": "get_kaggle_candidates", "description": "List downloadable weight candidates from a Kaggle model handle (e.g. 'google/albert/tensorFlow1/base/1'). Persists the candidate list to <repo_root>/.autodeploy/pending_weight_candidates.json.", "args": {"repo_root": Path, "kaggle_handle": str}},
        {"name": "kaggle_model_download", "description": "Download a Kaggle model by its handle (e.g. 'google/albert/tensorFlow1/base/1'). Downloads all model files into weights_related/.", "args": {"repo_root": Path, "kaggle_handle": str}},
        {"name": "get_zenodo_candidates", "description": "List downloadable weight candidates from a Zenodo record id or URL (e.g. 'https://zenodo.org/records/12345'). Persists the candidate list to <repo_root>/.autodeploy/pending_weight_candidates.json.", "args": {"repo_root": Path, "record_id": str}},
        {"name": "zenodo_weights_download", "description": "Download a selected weight file from Zenodo using record id and file key (filename).", "args": {"repo_root": Path, "record_id": str, "file_key": str}},
        {"name": "extract_archive", "description": "Extract a downloaded archive and return both the extraction directory and any discovered weight candidates.", "args": {"file_path": str}},
        {"name": "verify_weights_path", "description": "Verify that `path` points to a concrete weight file (suffix in .bin/.pt/.pth/.ckpt/.safetensors). This is the terminal tool of the phase.", "args": {"repo_root": Path, "path": str}},
        {"name": "update_weights_path", "description": "Manually set weights_path only when you already know the exact verified path.", "args": {"path": str}},
    ]
    allowed_parallel_phases: List[str] = ["dockersetup"]

    def __init__(self) -> None:
        super().__init__()
        self.backend = "gr"
        self.gpu_profile = resolve_gpu_profile()
        self.tools = {
            "get_file_tree": self.get_file_tree,
            "read_readme_content": self.read_readme_content,
            "read_text_resource": self.read_text_resource,
            "inspect_remote_weight_source": self.inspect_remote_weight_source,
            "find_local_weights": self.find_local_weights,
            "get_google_drive_candidates": self.get_google_drive_candidates,
            "get_huggingface_candidates": self.get_huggingface_candidates,
            "google_drive_weights_download": self.google_drive_weights_download,
            "huggingface_weights_download": self.huggingface_weights_download,
            "get_kaggle_candidates": self.get_kaggle_candidates,
            "kaggle_model_download": self.kaggle_model_download,
            "get_zenodo_candidates": self.get_zenodo_candidates,
            "zenodo_weights_download": self.zenodo_weights_download,
            "extract_archive": self.extract_archive,
            "verify_weights_path": self.verify_weights_path,
            "update_weights_path": self.update_weights_path,
        }
        prompt_path = os.path.join(project_path, "prompts")
        local_weights_prompt_filepath = os.path.join(prompt_path, "local_weights_prompt.json")
        with open(local_weights_prompt_filepath, "r", encoding="utf-8") as f:
            self.local_weights_prompt = json.load(f)

    def boundary_tools(self, tool_name: str) -> bool:
        # Only verify_weights_path is a true terminal state.
        # - find_local_weights is a *check*, not a terminal: when it succeeds it
        #   writes weights_path to the store and the LLM must still verify it.
        # - update_weights_path is a manual escape hatch with no validation, so
        #   marking it as a boundary would let the LLM short-circuit the phase
        #   by setting weights_path to anything.
        return tool_name == "verify_weights_path"

    def tool_arguments(self, tool_name: str) -> Dict[str, any]:
        tool_arguments_dict = {
            "get_file_tree": {"path": Path},
            "find_local_weights": {"repo_root": Path},
            "read_readme_content": {"cleaned_readmes_path": Path},
            "read_text_resource": {"repo_root": Path, "path": Path},
            "inspect_remote_weight_source": {"repo_root": Path, "source": str},
            "get_google_drive_candidates": {"repo_root": Path, "gd_url": str},
            "get_huggingface_candidates": {"repo_root": Path, "hf_repo_id": str},
            "google_drive_weights_download": {"repo_root": Path, "file_id": str},
            "huggingface_weights_download": {"repo_root": Path, "hf_repo_id": str, "file_path": str},
            "get_kaggle_candidates": {"repo_root": Path, "kaggle_handle": str},
            "kaggle_model_download": {"repo_root": Path, "kaggle_handle": str},
            "get_zenodo_candidates": {"repo_root": Path, "record_id": str},
            "zenodo_weights_download": {"repo_root": Path, "record_id": str, "file_key": str},
            "extract_archive": {"file_path": str},
            "verify_weights_path": {"repo_root": Path, "path": str},
            "update_weights_path": {"path": str},
        }
        return tool_arguments_dict[tool_name]

    def estimate_candidate_requirements(self, candidate_path: str, size_bytes: Optional[int]) -> Dict[str, Any]:
        path_lower = str(candidate_path).lower()
        file_size_gb = round(size_bytes / (1024 ** 3), 2) if size_bytes else None

        precision_bytes_map = {
            "fp32": 4.0,
            "float32": 4.0,
            "fp16": 2.0,
            "float16": 2.0,
            "bf16": 2.0,
            "int8": 1.0,
            "8bit": 1.0,
            "int4": 0.5,
            "4bit": 0.5,
            "q4": 0.5,
            "q5": 0.625,
            "q6": 0.75,
            "q8": 1.0,
        }
        detected_precision = None
        bytes_per_param = None
        for marker, value in precision_bytes_map.items():
            if marker in path_lower:
                detected_precision = marker
                bytes_per_param = value
                break

        param_match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)([bm])(?!\w)", path_lower)
        estimated_total_vram_gb = None
        if param_match and bytes_per_param is not None:
            param_value = float(param_match.group(1))
            unit = param_match.group(2)
            params_in_billions = param_value if unit == "b" else param_value / 1000.0
            estimated_total_vram_gb = round(params_in_billions * bytes_per_param * 1.15, 2)
        elif file_size_gb is not None:
            estimated_total_vram_gb = round(file_size_gb * 1.15, 2)

        return {
            "checkpoint_size_gb": file_size_gb,
            "detected_precision": detected_precision,
            "estimated_total_vram_gb": estimated_total_vram_gb,
        }

    def annotate_weight_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        profile = self.gpu_profile
        gpu_count = profile.get("gpu_count", 0)
        gpu_enabled = profile.get("gpu_enabled", False)
        max_per_gpu_memory_gb = profile.get("max_per_gpu_memory_gb")
        total_selected_memory_gb = profile.get("total_selected_memory_gb")

        annotated_candidates = []
        for candidate in candidates:
            candidate_copy = dict(candidate)
            resource_estimate = self.estimate_candidate_requirements(
                candidate_path=candidate_copy.get("path", ""),
                size_bytes=candidate_copy.get("size"),
            )
            candidate_copy.update(resource_estimate)
            candidate_copy["gpu_count"] = gpu_count
            candidate_copy["selected_gpu_indices"] = profile.get("selected_gpu_indices", [])
            candidate_copy["selected_gpu_memory_gb"] = profile.get("per_gpu_memory_gb", [])

            estimated_total_vram_gb = resource_estimate.get("estimated_total_vram_gb")
            if not gpu_enabled:
                fit_level = "gpu_not_enabled"
                fit_reason = "No GPU is enabled in config/global.yaml."
                fit_rank = 4
            elif estimated_total_vram_gb is None or max_per_gpu_memory_gb is None:
                fit_level = "unknown"
                fit_reason = "Unable to infer VRAM requirement from candidate name or local GPU inventory."
                fit_rank = 2
            elif estimated_total_vram_gb <= max_per_gpu_memory_gb:
                fit_level = "fits_single_gpu"
                fit_reason = f"Estimated VRAM {estimated_total_vram_gb} GB fits within one selected GPU."
                fit_rank = 0
            elif total_selected_memory_gb is not None and estimated_total_vram_gb <= total_selected_memory_gb:
                fit_level = "may_require_multi_gpu"
                fit_reason = f"Estimated VRAM {estimated_total_vram_gb} GB exceeds one GPU and may require multi-GPU loading."
                fit_rank = 1
            else:
                fit_level = "likely_too_large"
                fit_reason = f"Estimated VRAM {estimated_total_vram_gb} GB exceeds selected GPU memory."
                fit_rank = 3

            candidate_copy["gpu_fit_level"] = fit_level
            candidate_copy["gpu_fit_reason"] = fit_reason
            candidate_copy["selection_priority"] = fit_rank
            annotated_candidates.append(candidate_copy)

        annotated_candidates.sort(
            key=lambda item: (
                item.get("selection_priority", 99),
                -(item.get("size") or 0),
                item.get("path", ""),
            )
        )
        return annotated_candidates

    def get_file_tree(self, path: Path):
        tree = build_tree(path)
        return [{
            "value": tree,
            "storage": "permanent",
            "variable_name": f"file_tree of {path}"
        }]

    def read_readme_content(self, cleaned_readmes_path: Path):
        with open(cleaned_readmes_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        return [{
            "value": readme_content,
            "storage": "temporary",
            "variable_name": "readme_content"
        }]

    def assemble_local_weights_prompt(self, local_weights_candidates: List[dict]) -> str:
        prompt_parts = []
        
        role_prompt = "\n".join(self.local_weights_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)
        
        local_weights_candidates_prompt = "\n".join(self.local_weights_prompt['local_weights_candidates']).format(
            local_weights_candidates=json.dumps(local_weights_candidates)
        )
        prompt_parts.append("\n=== LOCAL WEIGHTS CANDIDATES ===")
        prompt_parts.append(local_weights_candidates_prompt)

        output_format_prompt = json.dumps(self.local_weights_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        return "\n".join(prompt_parts)

    def find_weight_files(self, tree, suffixes):
        results = []

        def dfs(node, current_path=""):
            path = os.path.join(current_path, node["name"])

            if node["type"] == "file" and node["name"].lower().endswith(tuple(suffixes)):
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

    def find_local_weights(self, repo_root: Path) -> List[Dict[str, Any]]:
        repo_tree = build_tree(repo_root)
        local_weights_candidates = self.find_weight_files(repo_tree, WEIGHT_SUFFIXES)
        if local_weights_candidates == []:
            return [{
                "value": "There is no local weights, please download the weights. You can NOT terminate the phase if there is no local weights.",
                "storage": "error",
                "variable_name": "no_local_weights"
            }]

        prompt = self.assemble_local_weights_prompt(local_weights_candidates)
        try:
            response = json_query(prompt, "local_weights_find", self.backend)
            print(response)
            if isinstance(response, str):
                response = json.loads(response)
            find_local_weights_flag = response['find_local_weights']
            if isinstance(find_local_weights_flag, str):
                find_local_weights_flag = find_local_weights_flag.strip().lower() == "true"
        except Exception as e:
            print(f"[find_local_weights] LLM query or parsing failed: {e}")
            return [{
                "value": f"find_local_weights failed to parse LLM response: {e}. Candidates found: {local_weights_candidates}",
                "storage": "error",
                "variable_name": "no_local_weights"
            }]

        if find_local_weights_flag:
            local_weights_path = response.get('local_weights_path', '')
            if not local_weights_path:
                return [{
                    "value": f"LLM claimed weights exist but returned no path. Candidates: {local_weights_candidates}",
                    "storage": "error",
                    "variable_name": "no_local_weights"
                }]
            src = Path(local_weights_path)
            if not src.is_absolute():
                src = repo_root / src
            src = src.resolve()
            if not src.exists():
                return [{
                    "value": f"LLM suggested weights path {src} does not exist on disk. Candidates: {local_weights_candidates}",
                    "storage": "error",
                    "variable_name": "no_local_weights"
                }]
            weights_dir = (repo_root / WEIGHTS_SUBDIR).resolve()
            # Copy to weights_related/ if not already there
            if not str(src).startswith(str(weights_dir)):
                weights_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                dst = weights_dir / src.name
                shutil.copy2(src, dst)
                print(f"[find_local_weights] copied {src} -> {dst}")
                local_weights_path = str(dst)
            return [{
                "value": local_weights_path,
                "storage": "permanent",
                "variable_name": "weights_path"
            }]
        else:
            return [{
                "value": "There is no local weights, please download the weights. You can NOT terminate the phase if there is no local weights.",
                "storage": "error",
                "variable_name": "no_local_weights"
            }]

    def _extract_google_drive_folder_id(self, source: str) -> str:
        if "/folders/" in source:
            return source.split("/folders/", 1)[1].split("?", 1)[0].split("/", 1)[0]
        return source.strip()

    def _extract_huggingface_repo_id(self, source: str) -> str:
        if "huggingface.co/" not in source:
            return source.strip().strip("/")
        tail = source.split("huggingface.co/", 1)[1].split("?", 1)[0].strip("/")
        parts = [part for part in tail.split("/") if part]
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return tail

    def normalize_path(self, repo_root: Path, path: Path):
        path = Path(path)

        # if it is already an absolute path, return as-is
        if path.is_absolute():
            return path

        parts = path.parts

        # if the first component is repo_root.name
        if parts and parts[0] == repo_root.name:
            path = Path(*parts[1:])

        return repo_root / path

    def read_text_resource(self, repo_root: Path, path: Path):
        if repo_root.parts[-1] in str(path):
            path = Path(path)
            path = str(path.relative_to(repo_root))
        else:
            path = str(path)
        found, current_path, node_type = locate_path(repo_root, path)
        print(found, current_path)
        if found:
            current_path = self.normalize_path(repo_root, current_path)
            with open(current_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                print(content)
            return [{
                "value": content, 
                "storage": "temporary", 
                "variable_name": f"content of {os.path.basename(current_path)}",
            }]
        return [{
            "value": "The local text resource is not found, check the path.",
            "storage": "error",
            "variable_name": "error_message"
        }]

    def inspect_remote_weight_source(self, repo_root: Path, source: str):
        source = str(source).strip()
        if "drive.google.com" in source:
            gd_url = source
            gd_repo_id = self._extract_google_drive_folder_id(source)
            candidates = self.google_drive_weights_candidates(gd_url)
            registry_path = _register_gd_candidates(repo_root, gd_url, candidates)
            return [
            {
                "value": candidates,
                "storage": "temporary",
                "variable_name": "gd_candidates",
            },
            {
                "value": gd_repo_id,
                "storage": "permanent",
                "variable_name": "gd_repo_id",
            },
            {
                "value": gd_url,
                "storage": "permanent",
                "variable_name": "gd_url",
            },
            {
                "value": str(registry_path),
                "storage": "permanent",
                "variable_name": "pending_weight_candidates_path",
            },
            ]
        elif "kaggle.com/models" in source:
            kaggle_handle = self._extract_kaggle_model_handle(source)
            candidates = self.kaggle_all_candidates(kaggle_handle)
            registry_path = _register_kaggle_candidates(repo_root, kaggle_handle, candidates)
            return [
            {
                "value": candidates,
                "storage": "temporary",
                "variable_name": "kaggle_candidates",
            },
            {
                "value": kaggle_handle,
                "storage": "permanent",
                "variable_name": "kaggle_handle",
            },
            {
                "value": source,
                "storage": "permanent",
                "variable_name": "kaggle_url",
            },
            {
                "value": str(registry_path),
                "storage": "permanent",
                "variable_name": "pending_weight_candidates_path",
            },
            ]
        elif "zenodo.org" in source:
            record_id = self._extract_zenodo_record_id(source)
            candidates = self.zenodo_all_candidates(record_id)
            registry_path = _register_zenodo_candidates(repo_root, record_id, candidates)
            return [
            {
                "value": candidates,
                "storage": "temporary",
                "variable_name": "zenodo_candidates",
            },
            {
                "value": record_id,
                "storage": "permanent",
                "variable_name": "zenodo_record_id",
            },
            {
                "value": source,
                "storage": "permanent",
                "variable_name": "zenodo_url",
            },
            {
                "value": str(registry_path),
                "storage": "permanent",
                "variable_name": "pending_weight_candidates_path",
            },
            ]
        elif "huggingface.co" in source or "/" in source:
            hf_repo_id = self._extract_huggingface_repo_id(source)
            candidates = self.huggingface_all_candidates(hf_repo_id)
            registry_path = _register_hf_candidates(repo_root, hf_repo_id, candidates)
            return [
            {
                "value": candidates,
                "storage": "temporary",
                "variable_name": "hf_candidates",
            },
            {
                "value": hf_repo_id,
                "storage": "permanent",
                "variable_name": "hf_repo_id",
            },
            {
                "value": source,
                "storage": "permanent",
                "variable_name": "hf_url",
            },
            {
                "value": str(registry_path),
                "storage": "permanent",
                "variable_name": "pending_weight_candidates_path",
            },
            ]
        else:
            return [{
                "value": "The remote source is not recognized. Provide a Google Drive folder URL, a Hugging Face repo id/URL, a Kaggle model URL, or a Zenodo record URL.",
                "storage": "error",
                "variable_name": "error_message"
            }]

    def list_drive_tree(self, folder_url_or_id: str, recursive: bool = True, _parent_path: str = "") -> Iterator[Union[DriveFile, DriveFolder]]:
        """
        Args:
            folder_url_or_id: Google Drive folder link or folder_id
            recursive: whether to recurse into subfolders
            _parent_path: internal; used to build the path during recursion

        Yields:
            DriveFile or DriveFolder
        """
        # extract folder_id
        if "drive.google.com" in folder_url_or_id:
            folder_id = folder_url_or_id.split("/folders/")[1].split("?")[0]
        else:
            folder_id = folder_url_or_id

        url = "https://www.googleapis.com/drive/v3/files"
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "key": GOOGLE_DRIVE_API_KEY,
            "fields": "nextPageToken,files(id,name,mimeType,size)",
            "pageSize": 1000,
        }

        while True:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            for f in data.get("files", []):
                path = f"{_parent_path}/{f['name']}" if _parent_path else f["name"]
                is_folder = f["mimeType"] == "application/vnd.google-apps.folder"

                if is_folder:
                    yield DriveFolder(path=path, id=f["id"])
                    if recursive:
                        yield from self.list_drive_tree(
                            f["id"], recursive=recursive, _parent_path=path
                        )
                else:
                    yield DriveFile(
                        path=path,
                        size=int(f.get("size", 0)),
                        id=f["id"],
                        mime_type=f["mimeType"],
                    )

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
            params["pageToken"] = next_page_token

    def google_drive_weights_candidates(self, gd_url: str) -> List[str]:

        weights_candidates = []

        for f in self.list_drive_tree(gd_url, recursive=True):
            if not isinstance(f, DriveFile):
                continue

            weights_candidates.append({
                "path": f.path,
                "size": f.size,
                "id": f.id,
            })

        return self.annotate_weight_candidates(weights_candidates)

    def get_google_drive_candidates(self, repo_root: Path, gd_url: str) -> str:
        candidates = self.google_drive_weights_candidates(gd_url)
        registry_path = _register_gd_candidates(repo_root, gd_url, candidates)
        return [{
            "value": candidates,
            "storage": "temporary",
            "variable_name": "gd_candidates",
        },
        {
            "value": str(registry_path),
            "storage": "permanent",
            "variable_name": "pending_weight_candidates_path",
        }]

    def get_huggingface_candidates(self, repo_root: Path, hf_repo_id: str) -> str:
        all_candidates = self.huggingface_all_candidates(hf_repo_id)
        registry_path = _register_hf_candidates(repo_root, hf_repo_id, all_candidates)
        return [{
            "value": all_candidates,
            "storage": "temporary",
            "variable_name": "hf_candidates",
        },
        {
            "value": str(registry_path),
            "storage": "permanent",
            "variable_name": "pending_weight_candidates_path",
        }]

    def _extract_kaggle_model_handle(self, source: str) -> str:
        # https://www.kaggle.com/models/google/albert/tensorFlow1/base/1?tfhub-redirect=true
        # -> "google/albert/tensorFlow1/base/1"
        if "kaggle.com/models/" not in source:
            return source.strip().strip("/")
        tail = source.split("kaggle.com/models/", 1)[1].split("?", 1)[0].strip("/")
        return tail

    def kaggle_all_candidates(self, kaggle_handle: str) -> List[Dict[str, Any]]:
        _setup_kaggle_credentials()
        import kagglehub
        cache_path = Path(kagglehub.model_download(kaggle_handle))
        candidates = []
        for f in cache_path.rglob("*"):
            if not f.is_file():
                continue
            name = f.name
            if name in self._HF_EXCLUDE_NAMES:
                continue
            suffix = f.suffix.lower()
            # Detect kind: standard weight suffixes, TF saved_model.pb,
            # and TF checkpoint data shards (.data-NNNNN-of-NNNNN)
            if suffix in WEIGHT_SUFFIXES:
                kind = "weight"
            elif name == "saved_model.pb" or re.search(r"\.data-\d{5}-of-\d{5}$", name):
                kind = "weight"
            else:
                kind = "auxiliary"
            size = f.stat().st_size
            entry = {
                "path": str(f.relative_to(cache_path)),
                "size": size,
                "suffix": suffix,
                "kind": kind,
            }
            candidates.append(entry)
        weight_entries = [c for c in candidates if c["kind"] == "weight"]
        if weight_entries:
            annotated = self.annotate_weight_candidates(weight_entries)
            annotated_by_path = {c["path"]: c for c in annotated}
            for c in candidates:
                if c["path"] in annotated_by_path:
                    c.update(annotated_by_path[c["path"]])
        return candidates

    def get_kaggle_candidates(self, repo_root: Path, kaggle_handle: str):
        candidates = self.kaggle_all_candidates(kaggle_handle)
        registry_path = _register_kaggle_candidates(repo_root, kaggle_handle, candidates)
        return [{
            "value": candidates,
            "storage": "temporary",
            "variable_name": "kaggle_candidates",
        },
        {
            "value": str(registry_path),
            "storage": "permanent",
            "variable_name": "pending_weight_candidates_path",
        }]

    def kaggle_model_download(self, repo_root: Path, kaggle_handle: str):
        try:
            target_dir = _download_kaggle_model(repo_root, kaggle_handle)
            weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
            # Remove all candidates for this handle from registry since we downloaded everything
            data = _load_registry(repo_root)
            if kaggle_handle in data.get("kaggle", {}):
                del data["kaggle"][kaggle_handle]
                _save_registry(repo_root, data)
            # Find the primary weight file to return, matching HF/GD behavior
            # Priority: WEIGHT_SUFFIXES files first, then saved_model.pb, then
            # TF checkpoint data shards (.data-NNNNN-of-NNNNN)
            weight_file = None
            tf_pb_file = None
            tf_data_shard = None
            for f in target_dir.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() in WEIGHT_SUFFIXES:
                    if weight_file is None or f.stat().st_size > weight_file.stat().st_size:
                        weight_file = f
                elif f.name == "saved_model.pb":
                    tf_pb_file = f
                elif re.search(r"\.data-\d{5}-of-\d{5}$", f.name):
                    if tf_data_shard is None or f.stat().st_size > tf_data_shard.stat().st_size:
                        tf_data_shard = f
            primary = weight_file or tf_pb_file or tf_data_shard
            downloaded_path = str(primary) if primary else str(target_dir)
            return [{
                "value": downloaded_path,
                "storage": "permanent",
                "variable_name": "downloaded_weight_path"
            },
            {
                "value": str(weights_dir),
                "storage": "permanent",
                "variable_name": "weights_dir"
            }]
        except Exception as e:
            return [{
                "value": f"Error downloading Kaggle model: {str(e)}",
                "storage": "error",
                "variable_name": "error_message"
            }]

    def _extract_zenodo_record_id(self, source: str) -> str:
        """Extract record id from Zenodo URL or raw id string.

        Supports:
            https://zenodo.org/records/12345
            https://zenodo.org/record/12345
            https://zenodo.org/api/records/12345
            12345
        """
        m = re.search(r'zenodo\.org/(?:api/)?records?/(\d+)', source)
        if m:
            return m.group(1)
        if source.strip().isdigit():
            return source.strip()
        raise ValueError(f"Cannot extract Zenodo record id from: {source}")

    def zenodo_all_candidates(self, record_id: str) -> List[Dict[str, Any]]:
        url = f"https://zenodo.org/api/records/{record_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        files = data.get("files", [])
        candidates = []
        for f in files:
            key = f.get("key", "")
            size = f.get("size", 0)
            download_url = f.get("links", {}).get("self", "")
            suffix = Path(key).suffix.lower()
            if suffix in WEIGHT_SUFFIXES:
                kind = "weight"
            elif suffix in AUXILIARY_SUFFIXES and key not in AUXILIARY_EXCLUDE_NAMES:
                kind = "auxiliary"
            elif suffix in {".zip", ".tar", ".gz", ".tgz", ".tar.gz"}:
                kind = "archive"
            else:
                kind = "other"
            entry = {
                "key": key,
                "size": size,
                "suffix": suffix,
                "kind": kind,
                "download_url": download_url,
            }
            candidates.append(entry)
        weight_entries = [c for c in candidates if c["kind"] == "weight"]
        if weight_entries:
            annotated = self.annotate_weight_candidates(
                [{"path": c["key"], "size": c["size"], "suffix": c["suffix"], "kind": c["kind"]} for c in weight_entries]
            )
            annotated_by_path = {c["path"]: c for c in annotated}
            for c in candidates:
                if c["key"] in annotated_by_path:
                    c.update({k: v for k, v in annotated_by_path[c["key"]].items() if k != "path"})
        return candidates

    def get_zenodo_candidates(self, repo_root: Path, record_id: str):
        record_id = self._extract_zenodo_record_id(str(record_id))
        candidates = self.zenodo_all_candidates(record_id)
        registry_path = _register_zenodo_candidates(repo_root, record_id, candidates)
        return [{
            "value": candidates,
            "storage": "temporary",
            "variable_name": "zenodo_candidates",
        },
        {
            "value": str(registry_path),
            "storage": "permanent",
            "variable_name": "pending_weight_candidates_path",
        }]

    def zenodo_weights_download(self, repo_root: Path, record_id: str, file_key: str):
        try:
            record_id = self._extract_zenodo_record_id(str(record_id))
            # Look up download_url from registry or fetch from API
            data = _load_registry(repo_root)
            download_url = None
            for c in data.get("zenodo", {}).get(record_id, []):
                if c.get("key") == file_key:
                    download_url = c.get("download_url")
                    break
            if not download_url:
                # Fallback: fetch from API
                candidates = self.zenodo_all_candidates(record_id)
                for c in candidates:
                    if c.get("key") == file_key:
                        download_url = c.get("download_url")
                        break
            if not download_url:
                return [{
                    "value": f"File '{file_key}' not found in Zenodo record {record_id}.",
                    "storage": "error",
                    "variable_name": "error_message"
                }]
            downloaded_path = _download_zenodo_file(repo_root, download_url, file_key)
            _remove_zenodo_candidate(repo_root, record_id, file_key)
            weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
            return [{
                "value": str(downloaded_path),
                "storage": "permanent",
                "variable_name": "downloaded_weight_path"
            },
            {
                "value": str(weights_dir),
                "storage": "permanent",
                "variable_name": "weights_dir"
            }]
        except Exception as e:
            return [{
                "value": f"Error downloading from Zenodo: {str(e)}",
                "storage": "error",
                "variable_name": "error_message"
            }]

    def google_drive_weights_download(self, repo_root: Path, file_id: str):
        try:
            downloaded_path = _download_google_drive_file(repo_root, file_id)
            _remove_gd_candidate(repo_root, file_id)
            weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
            return [{
                "value": str(downloaded_path),
                "storage": "permanent",
                "variable_name": "downloaded_weight_path"
            },
            {
                "value": str(weights_dir),
                "storage": "permanent",
                "variable_name": "weights_dir"
            }]
        except FileNotFoundError:
            return [{
                "value": f"File not found for Google Drive file_id: {file_id}",
                "storage": "error",
                "variable_name": "error_message"
            }]
        except Exception as e:
            return [{
                "value": f"Error downloading weights: {str(e)}",
                "storage": "error",
                "variable_name": "error_message"
            }]

    # Files that should never be registered as pending candidates —
    # they are repo metadata / docs, not inference artifacts.
    _HF_EXCLUDE_NAMES = {
        "README.md", ".gitattributes", ".gitignore",
        "LICENSE", "LICENSE.txt", "LICENSE.md",
        "NOTICE", "NOTICE.txt",
    }

    def huggingface_all_candidates(self, hf_repo_id: str) -> List[Dict[str, Any]]:
        """List ALL files in a HF repo as pending candidates, regardless of
        suffix. This ensures tokenizer.py, custom modeling code, sentencepiece
        models, and any other inference artifact are registered and can be
        downloaded later by ServiceDelivery if missing at runtime.
        """
        WEIGHT_SUFFIXES_SET = {".safetensors", ".bin", ".pt", ".pth"}
        repo_type = "model"
        candidates = []

        for f in list_repo_tree(hf_repo_id, repo_type=repo_type, recursive=True):
            if not isinstance(f, RepoFile):
                continue
            name = Path(f.path).name
            if name in self._HF_EXCLUDE_NAMES:
                continue
            suffix = Path(f.path).suffix.lower()
            kind = "weight" if suffix in WEIGHT_SUFFIXES_SET else "auxiliary"
            entry = {
                "path": f.path,
                "size": f.size,
                "suffix": suffix,
                "kind": kind,
            }
            candidates.append(entry)

        # Annotate weight candidates with GPU info
        weight_entries = [c for c in candidates if c["kind"] == "weight"]
        if weight_entries:
            annotated = self.annotate_weight_candidates(weight_entries)
            annotated_by_path = {c["path"]: c for c in annotated}
            for c in candidates:
                if c["path"] in annotated_by_path:
                    c.update(annotated_by_path[c["path"]])

        return candidates

    def huggingface_weights_download(self, repo_root: Path, hf_repo_id: str, file_path: str):
        try:
            weights_path = _download_huggingface_file(repo_root, hf_repo_id, file_path)
            _remove_hf_candidate(repo_root, hf_repo_id, file_path)
            weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
            return [{
                "value": str(weights_path),
                "storage": "permanent",
                "variable_name": "downloaded_weight_path"
            },
            {
                "value": str(weights_dir),
                "storage": "permanent",
                "variable_name": "weights_dir"
            }]
        except Exception as e:
            return [{
                "value": f"Error downloading weights: {str(e)}",
                "storage": "error",
                "variable_name": "error_message"
            }]

    def extract_archive(self, file_path: str):
        """
        Extract archive into the SAME directory.

        Parameters
        ----------
        file_path : str
            Archive path

        Returns
        -------
        str
            Extraction directory
        """

        file_path = Path(file_path)
        extract_dir = file_path.parent

        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as z:
                z.extractall(extract_dir)

        elif tarfile.is_tarfile(file_path):
            with tarfile.open(file_path, 'r:*') as t:
                t.extractall(extract_dir)

        else:
            raise ValueError(f"Unsupported archive: {file_path}")

        extract_tree = build_tree(extract_dir)
        extracted_weight_candidates = self.find_weight_files(extract_tree, WEIGHT_SUFFIXES)
        return [{
            "value": str(extract_dir),
            "storage": "temporary",
            "variable_name": "extract_dir"
        },
        {
            "value": extracted_weight_candidates,
            "storage": "temporary",
            "variable_name": "extracted_weight_candidates"
        }]

    def verify_weights_path(self, repo_root: Path, path: str):
        try:
            repo_root = Path(repo_root).resolve()
            p = Path(str(path).strip())
            if not p.is_absolute():
                parts = p.parts
                if parts and parts[0] == repo_root.name:
                    p = Path(*parts[1:])
                p = repo_root / p
            p = p.resolve()

            if not p.is_file():
                return [{
                    "value": f"weights path is not a file or does not exist: {p}",
                    "storage": "error",
                    "variable_name": "error_message"
                }]
            if p.suffix.lower() not in WEIGHT_SUFFIXES:
                return [{
                    "value": f"file is not a recognized weight file: {p}",
                    "storage": "error",
                    "variable_name": "error_message"
                }]
            weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
            # Copy to weights_related/ if not already there
            if not str(p).startswith(str(weights_dir)):
                weights_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                dst = weights_dir / p.name
                shutil.copy2(p, dst)
                print(f"[verify_weights_path] copied {p} -> {dst}")
                p = dst
            summary = (
                f"WeightResolve completed: weights_path={p}, "
                f"weights_dir={weights_dir}, weights_verified=True"
            )
            return [{
                "value": str(p),
                "storage": "permanent",
                "variable_name": "weights_path"
            },
            {
                "value": str(weights_dir),
                "storage": "permanent",
                "variable_name": "weights_dir"
            },
            {
                "value": summary,
                "storage": "temporary",
                "variable_name": "weightresolve_summary"
            }]
        except Exception as e:
            return [{
                "value": f"Error verifying weights path: {str(e)}",
                "storage": "error",
                "variable_name": "error_message"
            }]
    
    def update_weights_path(self, path: str):
        return [{
            "value": path,
            "storage": "permanent",
            "variable_name": "weights_path"
        }]
