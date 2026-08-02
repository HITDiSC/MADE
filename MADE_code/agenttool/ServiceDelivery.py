import json
import os
import difflib
import subprocess
import yaml
import time
import shutil
import uuid
import re
import urllib.request
from pathlib import Path
from typing import Any, List, Dict, Optional
from backend.query import query, json_query
from agenttool.base_phase import BasePhase
from agenttool.tool import linux_command, linux_command_in_docker, inspect_running, build_tree
from agenttool.FileTracker import FileTracker
from agenttool.gpu_lock import gpu_lock
# Reuse the same structural + consistency validators that adapt_code uses.
# This guarantees adapt_code and fix_runtime_code apply identical rules when
# accepting / rejecting an LLM-emitted input_schema, so the two co-generators
# never drift in their definition of "valid schema".
from agenttool.APIAdaptation import (
    _validate_input_schema_structure,
    _extract_raw_input_keys,
    _check_preprocess_schema_consistency,
    _schema_field_names,
)
from agenttool.WeightResolve import (
    WEIGHTS_SUBDIR,
    _load_registry,
    _download_google_drive_file,
    _download_huggingface_file,
    _download_kaggle_model,
    _download_zenodo_file,
    _remove_gd_candidate,
    _remove_hf_candidate,
    _remove_kaggle_candidate,
    _remove_zenodo_candidate,
)

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

# ── service.py function splitter / reassembler ──────────────────────────
import ast as _ast
from datetime import datetime

_PIPELINE_FUNCS = ("load_model", "preprocess", "inference", "postprocess")


def _split_service_code(code: str) -> Dict[str, str]:
    """Split service.py into its four pipeline functions + the rest.

    Returns a dict with keys: "header" (imports + globals before the first
    pipeline function), "load_model", "preprocess", "inference",
    "postprocess", and "footer" (init_model + predict boilerplate).
    Each value is the raw source text of that section, preserving newlines.
    """
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return {}

    lines = code.splitlines(keepends=True)
    # Collect (func_name, start_line_0based, end_line_0based) for pipeline funcs
    func_spans: List[tuple] = []
    for node in _ast.iter_child_nodes(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if node.name in _PIPELINE_FUNCS:
                start = node.lineno - 1  # 0-based
                end = node.end_lineno     # ast end_lineno is 1-based inclusive
                func_spans.append((node.name, start, end))

    if not func_spans:
        return {}

    # Sort by start line to handle any ordering
    func_spans.sort(key=lambda t: t[1])

    result: Dict[str, str] = {}
    # Header: everything before first pipeline function
    result["header"] = "".join(lines[:func_spans[0][1]])

    for i, (name, start, end) in enumerate(func_spans):
        result[name] = "".join(lines[start:end])

    # Footer: everything after the last pipeline function
    last_end = func_spans[-1][2]
    result["footer"] = "".join(lines[last_end:])

    return result


def _reassemble_service_code(parts: Dict[str, str]) -> str:
    """Inverse of _split_service_code: join parts back into a full file."""
    sections = [parts.get("header", "")]
    for func_name in _PIPELINE_FUNCS:
        if func_name in parts:
            sections.append(parts[func_name])
    sections.append(parts.get("footer", ""))
    return "\n".join(s.rstrip("\n") for s in sections if s) + "\n"


def _detect_error_function(api_log: str) -> Optional[str]:
    """Try to detect which pipeline function the error occurred in.

    Scans the traceback in the API log for 'in load_model', 'in preprocess',
    etc. Returns the function name or None if it can't be determined.
    """
    # Look for the innermost (last) pipeline function in the traceback
    found = None
    for line in api_log.splitlines():
        for func_name in _PIPELINE_FUNCS:
            if f"in {func_name}" in line:
                found = func_name
    return found


class ServiceDelivery(BasePhase):
    """Final phase of the pipeline: validate the deployed service end-to-end and ship.

    Symmetric to RepoIngest at the entry of the pipeline. Where RepoIngest takes
    raw paper input and produces a downstream-ready repo, ServiceDelivery takes
    the (built + adapted + container-ready) deployment and produces a verified
    running service plus the final delivery artifacts. It absorbs what used to
    be the RuntimeValidation + EndPhase pair, since those were a strict
    'validate -> document -> finalize' chain with no useful decision point in
    between for the phase manager to make.
    """

    name: str = "ServiceDelivery"
    description: str = (
        "Start the FastAPI inference service inside the Docker container, "
        "run all test cases to verify prediction correctness, "
        "fix runtime errors (code patching, package installation, "
        "downloading missing weight-related files such as tokenizer/config/vocab), "
        "generate API documentation, and finalize the deployment. "
        "Requires: container_id, fastapiapp_dir, service_pipeline_path, "
        "repo_root, test_file_dir, task, cleaned_readmes_path."
    )
    goal: str = (
        "Deliver a verified, running inference API that passes all test cases, "
        "together with its API documentation. Call end_deployment to finalize."
    )
    tools_schemas: List[Dict[str, any]] = [
        {"name": "check_container_status", "description": "Check whether the Docker container is running before trying to validate the API service. requires: container_id (from DockerSetUp). produces: container_status (bool). Call this first whenever any other tool in this phase fails or before you start a fresh validation cycle.", "args": {"container_id": str}},
        {"name": "docker_restart", "description": "Restart the Docker container when the runtime environment is unhealthy or the API process has exited. requires: container_id. produces: container_status (bool). Use this only after check_container_status reports the container is not running.", "args": {"container_id": str}},
        {"name": "get_container_logs", "description": "Read the tail of the uvicorn API log (repo_root/.autodeploy/uvicorn.log) when runtime validation fails and you need container-level diagnostics. requires: repo_root. produces: container_logs.", "args": {"repo_root": Path}},
        {"name": "install_package", "description": "Install a missing Python package inside the container when runtime validation identifies a dependency issue. Automatically resolves a compatible version from the Dockerfile so torch-family and other pinned packages stay ABI-compatible. requires: repo_root, package_name, container_id and a running container (check_container_status == true). produces: package_installed.", "args": {"repo_root": Path, "package_name": str, "container_id": str}},
        {"name": "uninstall_package", "description": "Uninstall a problematic Python package inside the container when runtime validation identifies a dependency conflict. requires: container_id and a running container. produces: package_uninstalled.", "args": {"package_name": str, "container_id": str}},
        {"name": "stop_api_service", "description": "Stop the uvicorn API process inside the container using a known api_pid when you need to clean up or retry validation. requires: container_id and api_pid (produced by validate_api_service). produces: api_service_stopped. Always call this before re-running validate_api_service so the new run starts from a clean process tree.", "args": {"container_id": str, "api_pid": str}},
        {"name": "list_pending_weight_candidates", "description": "Read <repo_root>/.autodeploy/pending_weight_candidates.json to see which Hugging Face, Google Drive, Kaggle, or Zenodo weight-related files were discovered by WeightResolve but have not been downloaded yet. IMPORTANT: when service_delivery_error_class is MISSING_FILE_ERROR, you MUST call this tool FIRST before calling fix_runtime_code — fix_runtime_code can only fix code paths, it cannot create files that do not exist. requires: repo_root. produces: pending_weight_candidates (dict with 'huggingface', 'google_drive', 'kaggle', and 'zenodo' buckets). If the registry is empty or the file is absent, no pending downloads remain.", "args": {"repo_root": Path}},
        {"name": "download_pending_huggingface_weight", "description": "Download a specific file from Hugging Face into <repo_root>/weights_related/ and remove it from the pending registry. IMPORTANT: when service_delivery_error_class is MISSING_FILE_ERROR, you MUST download the missing files BEFORE calling fix_runtime_code — the files must exist on disk first, then fix_runtime_code can update code paths to point to them. Use this after list_pending_weight_candidates identifies a missing HF file. requires: repo_root, hf_repo_id, file_path (both from the pending registry). produces: downloaded_weight_path, weights_dir. After downloading all needed files, call fix_runtime_code to correct paths in service.py, then re-run validate_api_service.", "args": {"repo_root": Path, "hf_repo_id": str, "file_path": str}},
        {"name": "download_pending_google_drive_weight", "description": "Download a specific file from Google Drive by file_id into <repo_root>/weights_related/ and remove it from the pending registry. IMPORTANT: when service_delivery_error_class is MISSING_FILE_ERROR, you MUST download the missing files BEFORE calling fix_runtime_code — the files must exist on disk first, then fix_runtime_code can update code paths to point to them. Use this after list_pending_weight_candidates identifies a missing Google Drive file. requires: repo_root, file_id (from the pending registry). produces: downloaded_weight_path, weights_dir. After downloading all needed files, call fix_runtime_code to correct paths in service.py, then re-run validate_api_service.", "args": {"repo_root": Path, "file_id": str}},
        {"name": "download_pending_kaggle_model", "description": "Download a Kaggle model by its handle into <repo_root>/weights_related/ and remove it from the pending registry. Use this after list_pending_weight_candidates identifies a missing Kaggle model. requires: repo_root, kaggle_handle (from the pending registry). produces: downloaded_weight_path, weights_dir.", "args": {"repo_root": Path, "kaggle_handle": str}},
        {"name": "download_pending_zenodo_weight", "description": "Download a specific file from Zenodo by record id and file key into <repo_root>/weights_related/ and remove it from the pending registry. Use this after list_pending_weight_candidates identifies a missing Zenodo file. requires: repo_root, record_id, file_key (from the pending registry). produces: downloaded_weight_path, weights_dir.", "args": {"repo_root": Path, "record_id": str, "file_key": str}},
        {"name": "fix_runtime_code", "description": "Fix runtime inference code errors by updating service.py to resolve either (a) a startup/crash failure — validate_api_service failed: use the service_delivery_error_class and service_delivery_error_suggestion variables it produced; or (b) an inference correctness failure — validate_inference_results returned validation_correctness_error: synthesize service_delivery_error_class='INFERENCE_LOGIC_ERROR', service_delivery_error_suggestion from the failed case details in validation_failures. The API log is read automatically from api_log_path. fix_target controls scope: 'auto' (default) detects the broken function from the traceback and fixes only that function; 'load_model'/'preprocess'/'inference'/'postprocess' fixes that specific function; 'all' rewrites the entire service.py. Prefer 'auto' or a specific function to avoid breaking working code. hint is a free-form string where you should describe your diagnosis and proposed fix direction based on what you have observed (e.g. 'The traceback shows barcodebert.model does not exist. The actual module is at barcodebert/barcodebert_model.py with class Barcodebert, fix the import path' or 'torchtext in the container is too old for disable_torchtext_deprecation_warning, remove that call'). Always provide a hint when you have useful context — it dramatically improves fix accuracy. reference_file_path is an optional path to a file in the repository whose content should be visible to the code-fix LLM as reference (e.g. the original inference script, a config file, or a model definition file). When the error involves config loading or wrong config values, pass the adapted config file path here so the fix LLM can see its content and adapt the code accordingly. Use get_file_tree to find the right file if unsure. Leave empty when not needed. IMPORTANT: when service_delivery_error_class is MISSING_FILE_ERROR, do NOT call this tool until you have first called list_pending_weight_candidates and downloaded the missing files via download_pending_huggingface_weight / download_pending_google_drive_weight / download_pending_kaggle_model / download_pending_zenodo_weight. This tool can only fix code paths, it cannot create missing files. Only modify service_pipeline_path; use fastapiapp_dir as reference context. After fixing, always call stop_api_service then validate_api_service again.", "args": {"repo_root": Path, "service_delivery_error_class": str, "service_delivery_error_suggestion": str, "service_pipeline_path": Path, "fix_target": str, "hint": str, "reference_file_path": str}},
        {"name": "fix_input_schema", "description": "Regenerate input_schema.json to match the preprocess() code, WITHOUT rewriting service.py. Use this when CONTRACT_ERROR is caused by a schema mismatch (field names, types, or required flags) but the preprocess code itself is correct. This is faster and more targeted than fix_runtime_code for pure schema issues. If this tool reports that the schema is already correct or inconsistent with preprocess code, fall back to fix_runtime_code with fix_target='preprocess'. After fixing, call stop_api_service then validate_api_service again.", "args": {"repo_root": Path, "service_pipeline_path": Path, "hint": str}},
        {"name": "validate_api_service", "description": "Start the FastAPI service inside the container, wait for the health endpoint, send a smoke-test prediction request, and report whether runtime inference succeeds. requires: container_id and a running container (call check_container_status first), test_file_dir, fastapiapp_dir. produces: api_pid, api_log_path, and on failure service_delivery_error_class / service_delivery_error_suggestion / service_delivery_api_log. This is NOT terminal - after success, call validate_inference_results next.", "args": {"repo_root": Path, "container_id": str, "test_file_dir": Path, "fastapiapp_dir": Path}},
        {"name": "validate_inference_results", "description": "Use the already running deployed API service to run inference on every test case under test_file_dir, then use an LLM to normalize API outputs and expected outputs into a shared field structure and compare each field. For each case, the harness uses a schema-aware matcher to map case input files to the fields declared in <repo_root>/.autodeploy/input_schema.json, then POSTs the composed JSON body to /predict_text. requires: repo_root (to locate input_schema.json), container_id, api_pid (produced by validate_api_service), test_file_dir. This is NOT terminal - after success, call generate_api_documentation, then end_deployment.", "args": {"repo_root": Path, "container_id": str, "api_pid": str, "test_file_dir": Path}},
        {"name": "generate_api_documentation", "description": "Generate the final API documentation markdown from the actual deployed service code (service.py) and FastAPI app code (app.py), grounded by the task description, the cleaned README, runtime info (container_id, api_pid, ports), input_schema, and one verified passing test case. Writes <repo_root>/.autodeploy/API_DOCUMENTATION.md. requires: repo_root, task, service_pipeline_path, fastapiapp_dir, cleaned_readmes_path, test_file_dir, container_id, api_pid. produces: api_documentation_path. Call this only after validate_inference_results has succeeded - the doc must reflect a verified working pipeline. Followed by end_deployment.", "args": {"repo_root": Path, "task": str, "service_pipeline_path": Path, "fastapiapp_dir": Path, "cleaned_readmes_path": Path, "test_file_dir": Path, "container_id": str, "api_pid": str}},
        {"name": "copy_local_to_weights", "description": "Copy a file or directory from the repository into <repo_root>/weights_related/. Use this when service_delivery_error_class is MISSING_FILE_ERROR and the missing file is NOT in the pending_weight_candidates registry (i.e. it is a local repo file, not a remote download). Typical use cases: config.json, tokenizer files (vocab.txt, tokenizer.model, tokenizer_config.json, special_tokens_map.json, spiece.model), label maps, or other auxiliary files that the model loading code expects inside weights_related/ but that live elsewhere in the repo. Workflow: (1) list_pending_weight_candidates shows the file is NOT pending for remote download, (2) use get_file_tree to locate the file in the repo, (3) use this tool to copy it into weights_related/, (4) call fix_runtime_code to update the path in service.py if needed. The source path (src_path) must be relative to repo_root or an absolute path inside the repo. requires: repo_root, src_path. produces: copied_path, weights_dir.", "args": {"repo_root": Path, "src_path": str}},
        {"name": "get_file_tree", "description": "Return the file tree for a directory path so you can inspect the repository structure, find reference files, or locate specific code files before calling fix_runtime_code with reference_file_path.", "args": {"path": Path}},
        {"name": "end_deployment", "description": "Mark the deployment as complete after runtime inference is validated and the API documentation has been written. This is the TERMINAL step of the phase and of the entire pipeline. requires: repo_root, api_pid, api_log_path, api_documentation_path. Re-publishes them to the variable store so downstream consumers (and any final report) can find them in one place.", "args": {"repo_root": Path, "api_pid": str, "api_log_path": Path, "api_documentation_path": Path}},
    ]
    allowed_parallel_phases: List[str] = []
    suggested_next_phases: List[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.backend = "gr"
        self.code_backend = "gr"
        self.tools = {
            "check_container_status": self.check_container_status,
            "docker_restart": self.docker_restart,
            "get_container_logs": self.get_container_logs,
            "install_package": self.install_package,
            "uninstall_package": self.uninstall_package,
            "stop_api_service": self.stop_api_service,
            "list_pending_weight_candidates": self.list_pending_weight_candidates,
            "download_pending_huggingface_weight": self.download_pending_huggingface_weight,
            "download_pending_google_drive_weight": self.download_pending_google_drive_weight,
            "download_pending_kaggle_model": self.download_pending_kaggle_model,
            "download_pending_zenodo_weight": self.download_pending_zenodo_weight,
            "fix_runtime_code": self.fix_runtime_code,
            "fix_input_schema": self.fix_input_schema,
            "copy_local_to_weights": self.copy_local_to_weights,
            "get_file_tree": self.get_file_tree,
            "validate_api_service": self.validate_api_service,
            "validate_inference_results": self.validate_inference_results,
            "generate_api_documentation": self.generate_api_documentation,
            "end_deployment": self.end_deployment,
        }
        prompt_path = os.path.join(project_path, "prompts")


        analyze_error_prompt_filepath = os.path.join(prompt_path, "analyze_error_prompt.json")
        with open(analyze_error_prompt_filepath, "r", encoding="utf-8") as f:
            self.analyze_error_prompt = json.load(f)

        runtime_codefix_prompt_filepath = os.path.join(prompt_path, "runtime_codefix_prompt.json")
        with open(runtime_codefix_prompt_filepath, "r", encoding="utf-8") as f:
            self.runtime_codefix_prompt = json.load(f)

        llm_normalize_prompt_filepath = os.path.join(prompt_path, "llm_normalize_prompt.json")
        with open(llm_normalize_prompt_filepath, "r", encoding="utf-8") as f:
            self.llm_normalize_prompt = json.load(f)

        api_doc_prompt_filepath = os.path.join(prompt_path, "endphase_api_doc_prompt.json")
        with open(api_doc_prompt_filepath, "r", encoding="utf-8") as f:
            self.api_doc_prompt = json.load(f)

        compose_request_body_prompt_filepath = os.path.join(prompt_path, "compose_request_body_prompt.json")
        with open(compose_request_body_prompt_filepath, "r", encoding="utf-8") as f:
            self.compose_request_body_prompt = json.load(f)

        cross_function_alignment_prompt_filepath = os.path.join(prompt_path, "cross_function_alignment_prompt.json")
        with open(cross_function_alignment_prompt_filepath, "r", encoding="utf-8") as f:
            self.cross_function_alignment_prompt = json.load(f)

    def boundary_tools(self, tool_name: str) -> bool:
        # end_deployment is the only true terminal: validation +
        # delivery artifact generation must both complete before the phase
        # exits. validate_api_service / validate_inference_results /
        # generate_api_documentation are stepping stones now.
        return tool_name == "end_deployment"

    def tool_arguments(self, tool_name: str) -> Dict[str, any]:
        tool_arguments_dict = {
            "check_container_status": {"container_id": str},
            "docker_restart": {"container_id": str},
            "get_container_logs": {"repo_root": Path},
            "install_package": {"repo_root": Path, "package_name": str, "container_id": str},
            "uninstall_package": {"package_name": str, "container_id": str},
            "stop_api_service": {"container_id": str, "api_pid": str},
            "list_pending_weight_candidates": {"repo_root": Path},
            "download_pending_huggingface_weight": {"repo_root": Path, "hf_repo_id": str, "file_path": str},
            "download_pending_google_drive_weight": {"repo_root": Path, "file_id": str},
            "download_pending_kaggle_model": {"repo_root": Path, "kaggle_handle": str},
            "download_pending_zenodo_weight": {"repo_root": Path, "record_id": str, "file_key": str},
            "fix_runtime_code": {"repo_root": Path, "service_delivery_error_class": str, "service_delivery_error_suggestion": str, "service_pipeline_path": Path, "fix_target": str, "hint": str, "reference_file_path": str},
            "fix_input_schema": {"repo_root": Path, "service_pipeline_path": Path, "hint": str},
            "copy_local_to_weights": {"repo_root": Path, "src_path": str},
            "get_file_tree": {"path": Path},
            "validate_api_service": {"repo_root": Path, "container_id": str, "test_file_dir": Path, "fastapiapp_dir": Path},
            "validate_inference_results": {"repo_root": Path, "container_id": str, "api_pid": str, "test_file_dir": Path},
            "generate_api_documentation": {"repo_root": Path, "task": str, "service_pipeline_path": Path, "fastapiapp_dir": Path, "cleaned_readmes_path": Path, "test_file_dir": Path, "container_id": str, "api_pid": str},
            "end_deployment": {"repo_root": Path, "api_pid": str, "api_log_path": Path, "api_documentation_path": Path},
        }
        return tool_arguments_dict[tool_name]

    def check_container_status(self, container_id: str):
        return [{
            "value": inspect_running(container_id),
            "storage": "temporary",
            "variable_name": "container_status"
        }]

    def docker_restart(self, container_id: str):
        try:
            restart_output = linux_command(f"docker restart {container_id}")
            return [{
                "value": restart_output.stdout.strip(),
                "storage": "temporary",
                "variable_name": "docker_restart_output"
            },
            {
                "value": inspect_running(container_id),
                "storage": "temporary",
                "variable_name": "container_status"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "docker_restart_error"
            }]

    def get_container_logs(self, repo_root: Path):
        try:
            log_path = Path(repo_root) / ".autodeploy" / "uvicorn.log"
            if not log_path.exists():
                return [{
                    "value": f"No uvicorn log found at {log_path}",
                    "storage": "error",
                    "variable_name": "container_logs_error"
                }]
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            # Only keep the last attempt block
            _sep = "--- new attempt "
            if _sep in content:
                content = content[content.rfind(_sep):]
            # Truncate to keep context manageable
            if len(content) > 4000:
                content = content[-4000:]
            return [{
                "value": content,
                "storage": "temporary",
                "variable_name": "container_logs"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "container_logs_error"
            }]

    def uninstall_package(self, package_name: str, container_id: str):
        try:
            linux_command_in_docker(f"python3 -m pip uninstall -y {package_name}", container_id)
            return [{
            "value": f"Successfully uninstalled {package_name}.",
            "storage": "temporary",
            "variable_name": "package_uninstalled"
        }]
        except Exception as e:
            return [{
                "value": f"Failed to uninstall {package_name}: {e}",
                "storage": "error",
                "variable_name": "package_uninstall_error"
            }]

    def install_package(self, repo_root: Path, package_name: str, container_id: str):
        try:
            # Read the Dockerfile to let a small LLM resolve a compatible
            # version (e.g. torchtext matching the torch already installed).
            dockerfile_path = Path(repo_root) / "Dockerfile"
            dockerfile_content = ""
            if dockerfile_path.exists():
                try:
                    dockerfile_content = dockerfile_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    dockerfile_content = ""

            # If no version is explicitly specified and we have a Dockerfile,
            # ask the LLM to pick a compatible version.
            versioned_package = package_name
            if "==" not in package_name and dockerfile_content:
                try:
                    prompt = (
                        "You are a Python dependency resolver. "
                        "Given a package name and a Dockerfile, determine the best compatible version to install.\n\n"
                        f"Package to install: {package_name}\n\n"
                        f"Dockerfile:\n{dockerfile_content}\n\n"
                        "Rules:\n"
                        "1. Check what Python packages and versions are already installed in the Dockerfile (look at pip install lines).\n"
                        "2. For torch-family packages (torchtext, torchvision, torchaudio, torchdata), "
                        "the version MUST match the torch version's release cycle. "
                        "For example, torch==2.0.1 pairs with torchtext==0.15.2, torchvision==0.15.2, torchaudio==2.0.2.\n"
                        "3. For other packages, pick a version compatible with the Python version and other deps in the Dockerfile.\n"
                        "4. If you cannot determine a specific version, return the package name without a version.\n"
                        "5. If the Dockerfile uses --index-url for torch wheels (e.g. cu118, cu121), "
                        "include the same --index-url in install_flags so the package is fetched from the matching channel.\n\n"
                        "Return JSON only:\n"
                        '{"package_spec": "<package_name>==<version> or just <package_name>", '
                        '"install_flags": "<extra pip flags like --index-url ... or empty string>", '
                        '"reason": "<brief explanation>"}'
                    )
                    resolve_result = json_query(prompt, "resolve_package_version", self.backend)
                    if isinstance(resolve_result, str):
                        resolve_result = json.loads(resolve_result)
                    versioned_package = resolve_result.get("package_spec", package_name)
                    install_flags = resolve_result.get("install_flags", "")
                    reason = resolve_result.get("reason", "")
                    print(
                        f"[ServiceDelivery] install_package: resolved {package_name} -> "
                        f"{versioned_package} (flags: {install_flags!r}, reason: {reason})"
                    )
                    if install_flags:
                        versioned_package = f"{versioned_package} {install_flags}"
                except Exception as e:
                    print(f"[ServiceDelivery] install_package: version resolve failed, using bare name: {e}")

            linux_command_in_docker(f"python3 -m pip install {versioned_package}", container_id)
            return [{
                "value": f"Successfully installed {versioned_package}.",
                "storage": "temporary",
                "variable_name": "package_installed"
            }]
        except Exception as e:
            return [{
                "value": f"Failed to install {package_name}: {e}",
                "storage": "error",
                "variable_name": "package_install_error"
            }]

    def stop_api_service(self, container_id: str, api_pid: str):
        try:
            # Graceful SIGTERM first so uvicorn can release the port cleanly.
            linux_command_in_docker(f"kill -15 {api_pid} 2>/dev/null || true", container_id)
            time.sleep(2)
            # Force-kill only if the process is still alive.
            linux_command_in_docker(
                f"kill -0 {api_pid} 2>/dev/null && kill -9 {api_pid} 2>/dev/null || true",
                container_id,
            )
            # Fallback: kill any remaining uvicorn processes, in case the
            # pid was wrong ('unknown') or uvicorn forked children that
            # outlived the parent.
            # Use [u]vicorn bracket trick to avoid pkill matching the
            # docker-exec bash wrapper (whose cmdline also contains
            # "uvicorn"), which would kill bash, make docker exec return
            # non-zero, and leave the real uvicorn alive.
            linux_command_in_docker(
                "pkill -9 -f '[u]vicorn' 2>/dev/null || true", container_id,
            )
            return [{
                "value": f"Stopped API process {api_pid}.",
                "storage": "temporary",
                "variable_name": "api_service_stopped"
            }]
        except Exception as e:
            return [{
                "value": f"Failed to stop API process {api_pid}: {e}",
                "storage": "error",
                "variable_name": "api_service_stop_error"
            }]

    def assemble_analyze_error_prompt(self, error_information: str):
        prompt_parts = []

        role_prompt = "\n".join(self.analyze_error_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        instructions_prompt = "\n".join(self.analyze_error_prompt['instructions'])
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append(instructions_prompt)

        error_information_prompt = self.analyze_error_prompt['error_information'].format(error_information=error_information)
        prompt_parts.append("\n=== ERROR INFORMATION ===")
        prompt_parts.append(error_information_prompt)

        output_format_prompt = "\n".join(self.analyze_error_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        prompt = "\n".join(prompt_parts)
        
        return prompt

    def assemble_runtime_code_fix_prompt(
        self,
        service_delivery_error_class: str,
        service_delivery_error_suggestion: str,
        service_delivery_api_log: str,
        service_pipeline_path: Path,
        service_pipeline_code: str,
        current_input_schema: str = "null",
        container_weights_dir: str = "",
        weights_dir_listing: str = "",
        hint: str = "",
        fix_target: str = "all",
        reference_file_path: str = "",
        reference_file_content: str = "",
    ) -> str:
        prompt_parts = []

        role_prompt = "\n".join(self.runtime_codefix_prompt["role"])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        instructions_prompt = "\n".join(self.runtime_codefix_prompt["instructions"])
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append(instructions_prompt)

        primary_target_prompt = "\n".join(self.runtime_codefix_prompt["primary_target"]).format(
            service_pipeline_path=service_pipeline_path,
        )
        prompt_parts.append("\n=== PRIMARY TARGET ===")
        prompt_parts.append(primary_target_prompt)

        runtime_error_prompt = "\n".join(self.runtime_codefix_prompt["runtime_error"]).format(
            service_delivery_error_class=service_delivery_error_class,
            service_delivery_error_suggestion=service_delivery_error_suggestion,
        )
        prompt_parts.append("\n=== RUNTIME ERROR ===")
        prompt_parts.append(runtime_error_prompt)

        if hint:
            prompt_parts.append("\n=== FIX HINT (from orchestrator) ===")
            prompt_parts.append(hint)

        if reference_file_content:
            prompt_parts.append(f"\n=== REFERENCE FILE ({reference_file_path}) ===")
            prompt_parts.append(reference_file_content)

        prompt_parts.append("\n=== API LOG ===")
        prompt_parts.append("\n".join(self.runtime_codefix_prompt["api_log"]).format(
            service_delivery_api_log=service_delivery_api_log
        ))

        prompt_parts.append("\n=== PRIMARY TARGET CODE ===")
        prompt_parts.append("\n".join(self.runtime_codefix_prompt["service_code"]).format(
            service_pipeline_code=service_pipeline_code
        ))

        if fix_target in ("preprocess", "all"):
            prompt_parts.append("\n=== CURRENT INPUT SCHEMA ===")
            prompt_parts.append("\n".join(self.runtime_codefix_prompt["current_input_schema"]).format(
                current_input_schema=current_input_schema
            ))
        else:
            prompt_parts.append("\n=== CURRENT INPUT SCHEMA ===")
            prompt_parts.append(
                "Schema is managed separately and is NOT relevant to this fix target. "
                "Do NOT modify or emit a new schema. Set schema_changed=false and fixed_input_schema=null."
            )

        if fix_target in ("load_model", "all"):
            prompt_parts.append("\n=== AVAILABLE WEIGHT FILES ===")
            prompt_parts.append("\n".join(self.runtime_codefix_prompt["available_weight_files"]).format(
                container_weights_dir=container_weights_dir,
                weights_dir_listing=weights_dir_listing,
            ))

        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append("\n".join(self.runtime_codefix_prompt["rules"]))

        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(json.dumps(self.runtime_codefix_prompt["output_format"]))

        return "\n".join(prompt_parts)

    def copy_local_to_weights(self, repo_root: Path, src_path: str) -> List[Dict[str, Any]]:
        """Copy a local file or directory from the repo into weights_related/."""
        try:
            repo_root = Path(repo_root).resolve()
            weights_dir = repo_root / WEIGHTS_SUBDIR
            weights_dir.mkdir(parents=True, exist_ok=True)

            src = Path(src_path)
            if not src.is_absolute():
                src = (repo_root / src).resolve()
            else:
                src = src.resolve()

            if not src.exists():
                return [{
                    "value": f"Source path does not exist: {src}",
                    "storage": "error",
                    "variable_name": "copy_local_error"
                }]

            # Prevent copying weights_related into itself
            if str(src).startswith(str(weights_dir)):
                return [{
                    "value": f"Source is already inside weights_related/: {src}",
                    "storage": "temporary",
                    "variable_name": "copy_local_skipped"
                }]

            dst = weights_dir / src.name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

            print(f"[ServiceDelivery] copied {src} -> {dst}")
            return [{
                "value": str(dst),
                "storage": "temporary",
                "variable_name": "copied_path"
            },
            {
                "value": str(weights_dir),
                "storage": "temporary",
                "variable_name": "weights_dir"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "copy_local_error"
            }]

    def get_file_tree(self, path: Path):
        tree = build_tree(path)
        return [{
            "value": tree,
            "storage": "temporary",
            "variable_name": f"file_tree of {path}"
        }]

    def list_pending_weight_candidates(self, repo_root: Path):
        try:
            registry = _load_registry(Path(repo_root))
            return [{
                "value": registry,
                "storage": "temporary",
                "variable_name": "pending_weight_candidates",
            }]
        except Exception as e:
            return [{
                "value": f"Error reading pending weight candidates: {str(e)}",
                "storage": "error",
                "variable_name": "pending_weight_candidates_error",
            }]

    def download_pending_huggingface_weight(self, repo_root: Path, hf_repo_id: str, file_path: str):
        try:
            downloaded = _download_huggingface_file(Path(repo_root), hf_repo_id, file_path)
            _remove_hf_candidate(Path(repo_root), hf_repo_id, file_path)
            weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
            return [{
                "value": str(downloaded),
                "storage": "permanent",
                "variable_name": "downloaded_weight_path",
            },
            {
                "value": str(weights_dir),
                "storage": "permanent",
                "variable_name": "weights_dir",
            }]
        except Exception as e:
            return [{
                "value": f"Error downloading Hugging Face weight {hf_repo_id}:{file_path}: {str(e)}",
                "storage": "error",
                "variable_name": "weight_download_error",
            }]

    def download_pending_google_drive_weight(self, repo_root: Path, file_id: str):
        try:
            downloaded = _download_google_drive_file(Path(repo_root), file_id)
            _remove_gd_candidate(Path(repo_root), file_id)
            weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
            return [{
                "value": str(downloaded),
                "storage": "permanent",
                "variable_name": "downloaded_weight_path",
            },
            {
                "value": str(weights_dir),
                "storage": "permanent",
                "variable_name": "weights_dir",
            }]
        except Exception as e:
            return [{
                "value": f"Error downloading Google Drive file_id {file_id}: {str(e)}",
                "storage": "error",
                "variable_name": "weight_download_error",
            }]

    def download_pending_kaggle_model(self, repo_root: Path, kaggle_handle: str):
        try:
            import re as _re
            downloaded_dir = _download_kaggle_model(Path(repo_root), kaggle_handle)
            # Clear all candidates for this handle
            from agenttool.WeightResolve import _load_registry as _lr, _save_registry as _sr, WEIGHT_SUFFIXES as _WS
            data = _lr(Path(repo_root))
            if kaggle_handle in data.get("kaggle", {}):
                del data["kaggle"][kaggle_handle]
                _sr(Path(repo_root), data)
            weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
            # Find the primary weight file, matching WeightResolve.kaggle_model_download
            weight_file = None
            tf_pb_file = None
            tf_data_shard = None
            for f in downloaded_dir.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() in _WS:
                    if weight_file is None or f.stat().st_size > weight_file.stat().st_size:
                        weight_file = f
                elif f.name == "saved_model.pb":
                    tf_pb_file = f
                elif _re.search(r"\.data-\d{5}-of-\d{5}$", f.name):
                    if tf_data_shard is None or f.stat().st_size > tf_data_shard.stat().st_size:
                        tf_data_shard = f
            primary = weight_file or tf_pb_file or tf_data_shard
            downloaded_path = str(primary) if primary else str(downloaded_dir)
            return [{
                "value": downloaded_path,
                "storage": "permanent",
                "variable_name": "downloaded_weight_path",
            },
            {
                "value": str(weights_dir),
                "storage": "permanent",
                "variable_name": "weights_dir",
            }]
        except Exception as e:
            return [{
                "value": f"Error downloading Kaggle model {kaggle_handle}: {str(e)}",
                "storage": "error",
                "variable_name": "weight_download_error",
            }]

    def download_pending_zenodo_weight(self, repo_root: Path, record_id: str, file_key: str):
        try:
            # Look up download_url from registry
            data = _load_registry(Path(repo_root))
            download_url = None
            for c in data.get("zenodo", {}).get(record_id, []):
                if c.get("key") == file_key:
                    download_url = c.get("download_url")
                    break
            if not download_url:
                # Fallback: construct URL from Zenodo API convention
                download_url = f"https://zenodo.org/api/records/{record_id}/files/{file_key}/content"
            downloaded = _download_zenodo_file(Path(repo_root), download_url, file_key)
            _remove_zenodo_candidate(Path(repo_root), record_id, file_key)
            weights_dir = (Path(repo_root) / WEIGHTS_SUBDIR).resolve()
            return [{
                "value": str(downloaded),
                "storage": "permanent",
                "variable_name": "downloaded_weight_path",
            },
            {
                "value": str(weights_dir),
                "storage": "permanent",
                "variable_name": "weights_dir",
            }]
        except Exception as e:
            return [{
                "value": f"Error downloading Zenodo file {record_id}/{file_key}: {str(e)}",
                "storage": "error",
                "variable_name": "weight_download_error",
            }]

    def fix_runtime_code(self, repo_root: Path, service_delivery_error_class: str, service_delivery_error_suggestion: str, service_pipeline_path: Path, fix_target: str = "auto", hint: str = "", reference_file_path: str = ""):
        try:
            repo_root = Path(repo_root).resolve()
            service_pipeline_path = Path(service_pipeline_path).resolve()

            service_pipeline_code = service_pipeline_path.read_text(encoding="utf-8", errors="ignore")

            # Read API log directly from disk instead of receiving it as a
            # parameter. This keeps the raw log out of the agent's context
            # (error_suggestion already has the distilled analysis) while
            # still giving the code-fix LLM the full traceback it needs.
            api_log_path = repo_root / ".autodeploy" / "uvicorn.log"
            try:
                service_delivery_api_log = api_log_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                service_delivery_api_log = ""
            # Only keep the last attempt block so the code-fix LLM
            # focuses on the current error, not stale history.
            _ATTEMPT_SEP = "--- new attempt "
            if _ATTEMPT_SEP in service_delivery_api_log:
                last_idx = service_delivery_api_log.rfind(_ATTEMPT_SEP)
                service_delivery_api_log = service_delivery_api_log[last_idx:]
            if len(service_delivery_api_log) > 4000:
                service_delivery_api_log = (
                    service_delivery_api_log[:2000]
                    + "\n...[truncated middle]...\n"
                    + service_delivery_api_log[-2000:]
                )

            # Read reference file if provided by EM
            reference_file_content = ""
            if reference_file_path:
                ref_path = Path(reference_file_path)
                if not ref_path.is_absolute():
                    ref_path = repo_root / ref_path
                try:
                    reference_file_content = ref_path.read_text(encoding="utf-8", errors="ignore")
                    print(f"[fix_runtime_code] loaded reference file: {ref_path} ({len(reference_file_content)} chars)")
                except Exception as e:
                    print(f"[fix_runtime_code] failed to read reference file {ref_path}: {e}")

            file_tracker = FileTracker(str(repo_root))
            file_tracker.init_workspace()

            # Current schema (included in prompt only when fix_target is preprocess/all).
            schema_path = repo_root / ".autodeploy" / "input_schema.json"
            current_schema = None
            if schema_path.exists():
                try:
                    current_schema = json.loads(schema_path.read_text(encoding="utf-8"))
                except Exception:
                    current_schema = None
            current_schema_text = json.dumps(current_schema, ensure_ascii=False, indent=2) if current_schema is not None else "null"

            # Scan weights_related directory so the LLM knows what files
            # are available inside the container for model loading.
            weights_dir = repo_root / WEIGHTS_SUBDIR
            work_dir = str(docker_setting.get("work_dir") or "/workspace")
            container_weights_dir = f"{work_dir}/{WEIGHTS_SUBDIR}"
            if weights_dir.is_dir():
                entries = sorted(
                    str(p.relative_to(weights_dir))
                    for p in weights_dir.rglob("*") if p.is_file()
                )
                weights_dir_listing = "\n".join(f"- {e}" for e in entries) if entries else "(empty)"
            else:
                weights_dir_listing = "(directory does not exist)"

            # ─── Targeted fix: split service.py and only fix the broken function ──
            targeted_func = None  # None means "fix all" (legacy behavior)
            code_parts = {}
            if fix_target != "all":
                code_parts = _split_service_code(service_pipeline_code)
                if code_parts:
                    if fix_target == "auto":
                        targeted_func = _detect_error_function(service_delivery_api_log)
                        if targeted_func and targeted_func not in code_parts:
                            targeted_func = None  # fallback to all
                    elif fix_target in _PIPELINE_FUNCS and fix_target in code_parts:
                        targeted_func = fix_target
                    # else: invalid fix_target or parse failure → fall through to all

                    if targeted_func:
                        print(
                            f"[fix_runtime_code] targeted mode: fixing only "
                            f"'{targeted_func}' (fix_target={fix_target!r})"
                        )
                        # Send only the target function but include the full file
                        # as read-only context so the LLM can see imports, model
                        # bundle shape, etc. Mark clearly which part to fix.
                        service_pipeline_code_for_prompt = (
                            f"=== FULL SERVICE.PY (read-only context) ===\n"
                            f"{service_pipeline_code}\n\n"
                            f"=== FUNCTION TO FIX (return ONLY this function's code) ===\n"
                            f"{code_parts[targeted_func]}"
                        )
                    else:
                        print(
                            f"[fix_runtime_code] could not isolate target function "
                            f"(fix_target={fix_target!r}), falling back to full rewrite"
                        )
                        service_pipeline_code_for_prompt = service_pipeline_code
                else:
                    print("[fix_runtime_code] AST split failed, falling back to full rewrite")
                    service_pipeline_code_for_prompt = service_pipeline_code
            else:
                service_pipeline_code_for_prompt = service_pipeline_code

            prompt = self.assemble_runtime_code_fix_prompt(
                service_delivery_error_class=service_delivery_error_class,
                service_delivery_error_suggestion=service_delivery_error_suggestion,
                service_delivery_api_log=service_delivery_api_log,
                service_pipeline_path=service_pipeline_path,
                service_pipeline_code=service_pipeline_code_for_prompt,
                current_input_schema=current_schema_text,
                container_weights_dir=container_weights_dir,
                weights_dir_listing=weights_dir_listing,
                hint=hint,
                fix_target=targeted_func if targeted_func else "all",
                reference_file_path=reference_file_path,
                reference_file_content=reference_file_content,
            )

            fix_response = query(prompt, self.code_backend)
            if isinstance(fix_response, str):
                fix_response = json.loads(fix_response)

            fixed_code_text     = fix_response.get("fixed_code_text", "")
            fixed_input_schema  = fix_response.get("fixed_input_schema")
            schema_changed      = bool(fix_response.get("schema_changed", False))
            fix_reason          = fix_response.get("reason", "")

            # If we were in targeted mode, splice the fixed function back
            # into the full service.py.
            if targeted_func and code_parts:
                code_parts[targeted_func] = fixed_code_text
                fixed_code_text = _reassemble_service_code(code_parts)
                print(
                    f"[fix_runtime_code] spliced fixed '{targeted_func}' back into "
                    f"full service.py ({len(fixed_code_text)} chars)"
                )

                # ─── Post-validation 0: post-fix review ─────────────────
                # After splicing a single fixed function back, do a holistic
                # review of the full service.py: does the targeted fix
                # actually solve the original error? Are the four functions
                # still compatible? Are there obvious bugs in other functions?
                # If the review says a full rewrite is needed, reject and
                # tell EM to use fix_target="all".
                original_error = (
                    f"error_class: {service_delivery_error_class}\n"
                    f"error_suggestion: {service_delivery_error_suggestion}\n"
                    f"api_log (tail):\n{service_delivery_api_log}"
                )
                review_issues = self._verify_cross_function_alignment(
                    fixed_code_text, targeted_func, original_error
                )
                if review_issues:
                    error_detail = "\n".join(f"  - {e}" for e in review_issues)
                    return [{
                        "value": (
                            f"fix_runtime_code rejected: post-fix review determined that "
                            f"the targeted fix to '{targeted_func}' is insufficient:\n"
                            f"{error_detail}\n\n"
                            f"Retry with fix_target=\"all\" to rewrite the entire "
                            f"service.py so all four functions stay consistent."
                        ),
                        "storage": "error",
                        "variable_name": "runtime_fix_error",
                    }]

            # ─── Post-validation 1: downgrade "changed but identical" ────────
            # If the LLM claims schema_changed=true but the emitted schema is
            # byte-for-byte the same as what's on disk, treat it as a no-op so
            # we don't waste a FileTracker version bump.
            if schema_changed and fixed_input_schema is not None and current_schema is not None:
                if json.dumps(fixed_input_schema, sort_keys=True) == json.dumps(current_schema, sort_keys=True):
                    print(
                        "[fix_runtime_code] LLM set schema_changed=true but new "
                        "schema is identical to current; downgrading to false."
                    )
                    schema_changed = False
                    fixed_input_schema = None

            # ─── Post-validation 2: reject silent code-side drift ────────────
            # If the LLM claims schema_changed=false but the new code reads
            # raw_input[...] keys that differ from the current schema, the
            # LLM has silently changed the contract without declaring it.
            # Reject the fix and force a retry.
            new_code_keys = _extract_raw_input_keys(fixed_code_text)
            # current_schema uses list form: {"inputs": [{"field_name": "...", ...}, ...]}
            current_schema_keys = set()
            if isinstance(current_schema, dict):
                inputs_list = current_schema.get("inputs") or []
                if isinstance(inputs_list, list):
                    current_schema_keys = {
                        item["field_name"] for item in inputs_list
                        if isinstance(item, dict) and isinstance(item.get("field_name"), str)
                    }

            if not schema_changed and current_schema is not None and new_code_keys != current_schema_keys:
                if not new_code_keys:
                    detail = (
                        f"fix_runtime_code rejected: LLM set schema_changed=false but "
                        f"the fixed preprocess() contains ZERO raw_input[...] accesses "
                        f"(expected {sorted(current_schema_keys)}). "
                        f"The preprocess function MUST read its inputs via "
                        f"raw_input['<field_name>'] — each value is a file path. "
                        f"Do NOT rename the parameter, use **kwargs, or access inputs "
                        f"by any other pattern. Re-emit preprocess so it accesses "
                        + ", ".join(f"raw_input['{k}']" for k in sorted(current_schema_keys))
                        + " (or set schema_changed=true and emit a new fixed_input_schema "
                        f"if the field set genuinely needs to change)."
                    )
                else:
                    detail = (
                        f"fix_runtime_code rejected: LLM set schema_changed=false but "
                        f"the fixed code reads raw_input[...] keys "
                        f"{sorted(new_code_keys)} which differ from the current "
                        f"schema keys {sorted(current_schema_keys)}. The LLM must "
                        f"either emit a new fixed_input_schema (with schema_changed=true) "
                        f"or leave the code's raw_input[...] accesses alone."
                    )
                return [{
                    "value": detail,
                    "storage": "error",
                    "variable_name": "runtime_fix_error",
                }]

            # ─── Post-validation 3: when schema IS changed, run structural +
            # consistency checks the same way adapt_code does. This is what
            # guarantees the two co-generators apply identical rules.
            if schema_changed and fixed_input_schema is not None:
                structural_errors = _validate_input_schema_structure(fixed_input_schema)
                if structural_errors:
                    return [{
                        "value": (
                            "fix_runtime_code rejected: new input_schema is structurally invalid:\n"
                            + "\n".join(f"  - {e}" for e in structural_errors)
                        ),
                        "storage": "error",
                        "variable_name": "runtime_fix_error",
                    }]

                _, consistency_errors = _check_preprocess_schema_consistency(
                    fixed_code_text, fixed_input_schema
                )
                if consistency_errors:
                    declared = _schema_field_names(fixed_input_schema)
                    return [{
                        "value": (
                            "fix_runtime_code rejected: new code/schema are inconsistent:\n"
                            + "\n".join(f"  - {e}" for e in consistency_errors)
                            + f"\n  preprocess reads raw_input[...] keys: {sorted(new_code_keys)}"
                            + f"\n  new schema declares field_names:       {sorted(declared)}"
                        ),
                        "storage": "error",
                        "variable_name": "runtime_fix_error",
                    }]

            # ─── No-op fast path: code unchanged, schema unchanged ──────────
            if fixed_code_text == service_pipeline_code and not schema_changed:
                return [{
                    "value": [],
                    "storage": "temporary",
                    "variable_name": "runtime_fix_modified_files"
                },
                {
                    "value": fix_reason,
                    "storage": "temporary",
                    "variable_name": "runtime_fix_summary"
                },
                {
                    "value": False,
                    "storage": "temporary",
                    "variable_name": "runtime_fix_applied_flag"
                }]

            modified_files = []
            observations = []

            # Write service.py only if it actually changed
            if fixed_code_text != service_pipeline_code:
                new_code_version = file_tracker.modify_file(str(service_pipeline_path), fixed_code_text)
                modified_files.append(str(service_pipeline_path))
                observations.append({
                    "value": {"file_path": str(service_pipeline_path), "version": new_code_version},
                    "storage": "temporary",
                    "variable_name": "runtime_fix_version"
                })

            # Write input_schema.json only when schema_changed=true
            if schema_changed and fixed_input_schema is not None:
                schema_path.parent.mkdir(parents=True, exist_ok=True)
                new_schema_version = file_tracker.modify_file(
                    str(schema_path),
                    json.dumps(fixed_input_schema, ensure_ascii=False, indent=2),
                )
                modified_files.append(str(schema_path))
                observations.append({
                    "value": {"file_path": str(schema_path), "version": new_schema_version},
                    "storage": "temporary",
                    "variable_name": "runtime_fix_schema_version"
                })
                observations.append({
                    "value": fixed_input_schema,
                    "storage": "permanent",
                    "variable_name": "input_schema"
                })
                print(f"[fix_runtime_code] schema updated to FileTracker v{new_schema_version}")

            observations.insert(0, {
                "value": modified_files,
                "storage": "temporary",
                "variable_name": "runtime_fix_modified_files"
            })
            observations.append({
                "value": fix_reason,
                "storage": "temporary",
                "variable_name": "runtime_fix_summary"
            })
            observations.append({
                "value": True,
                "storage": "temporary",
                "variable_name": "runtime_fix_applied_flag"
            })
            return observations
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "runtime_fix_error"
            }]

    # ──────────────────────────────────────────────────────────────────────
    # fix_input_schema — lightweight schema-only fix for CONTRACT_ERROR
    # ──────────────────────────────────────────────────────────────────────

    def fix_input_schema(
        self,
        repo_root: Path,
        service_pipeline_path: Path,
        hint: str = "",
    ) -> List[Dict[str, Any]]:
        """Regenerate input_schema.json to match the preprocess code,
        WITHOUT rewriting service.py.

        Use this when CONTRACT_ERROR is caused by a schema mismatch (field
        names, types, or required flags don't match the preprocess code)
        but the preprocess code itself is correct.
        """
        try:
            repo_root = Path(repo_root).resolve()
            service_pipeline_path = Path(service_pipeline_path).resolve()

            # Read current service.py
            service_code = service_pipeline_path.read_text(encoding="utf-8", errors="ignore")

            # Extract preprocess function code
            code_parts = _split_service_code(service_code)
            preprocess_code = code_parts.get("preprocess", "") if code_parts else ""
            if not preprocess_code:
                return [{
                    "value": "Could not extract preprocess() from service.py. Use fix_runtime_code with fix_target='all' instead.",
                    "storage": "error",
                    "variable_name": "fix_input_schema_error",
                }]

            # Read current schema
            schema_path = repo_root / ".autodeploy" / "input_schema.json"
            current_schema = None
            if schema_path.exists():
                try:
                    current_schema = json.loads(schema_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            current_schema_text = json.dumps(current_schema, ensure_ascii=False, indent=2) if current_schema else "null"

            # Build the LLM prompt
            prompt_parts = [
                "=== ROLE ===",
                "You are an input schema specialist. Your ONLY job is to produce a corrected input_schema.json that is consistent with the preprocess() code.",
                "",
                "=== PREPROCESS CODE ===",
                preprocess_code,
                "",
                "=== CURRENT INPUT SCHEMA ===",
                current_schema_text,
                "",
            ]
            if hint:
                prompt_parts.extend([
                    "=== HINT ===",
                    hint,
                    "",
                ])
            prompt_parts.extend([
                "=== RULES ===",
                "1. Look at raw_input['...'] accesses in the preprocess code to determine the field names.",
                "2. field_name MUST match ^[a-z_][a-z0-9_]*$ (snake_case).",
                "3. field_type must be one of: string, number, integer, boolean.",
                "4. field_required is a bool.",
                "5. field_description is one sentence.",
                "6. The set of field_names MUST exactly match the raw_input keys used in preprocess code.",
                "7. Do NOT add fields that preprocess doesn't read. Do NOT omit fields that preprocess reads.",
                "",
                "=== OUTPUT FORMAT ===",
                json.dumps({
                    "reason": "<why you chose these fields>",
                    "input_schema": {
                        "reason": "<inference audit trail>",
                        "version": "1.0",
                        "task_type": "<e.g. text_generation, extractive_qa, classification>",
                        "inputs": [
                            {
                                "field_name": "<snake_case>",
                                "field_type": "string",
                                "field_required": True,
                                "field_description": "<one sentence>"
                            }
                        ]
                    }
                }),
            ])
            prompt = "\n".join(prompt_parts)

            response = json_query(prompt, "fix_input_schema", self.code_backend)
            if isinstance(response, str):
                response = json.loads(response)

            new_schema = response.get("input_schema")
            if new_schema is None:
                return [{
                    "value": "LLM did not emit input_schema in its response.",
                    "storage": "error",
                    "variable_name": "fix_input_schema_error",
                }]

            # Validate structure
            structural_errors = _validate_input_schema_structure(new_schema)
            if structural_errors:
                error_detail = "\n".join(f"  - {e}" for e in structural_errors)
                return [{
                    "value": f"fix_input_schema rejected: new schema is structurally invalid:\n{error_detail}",
                    "storage": "error",
                    "variable_name": "fix_input_schema_error",
                }]

            # Validate consistency with preprocess code
            _, consistency_errors = _check_preprocess_schema_consistency(
                preprocess_code, new_schema
            )
            if consistency_errors:
                error_detail = "\n".join(f"  - {e}" for e in consistency_errors)
                return [{
                    "value": (
                        f"fix_input_schema rejected: new schema is inconsistent with "
                        f"preprocess code:\n{error_detail}\n\n"
                        f"Use fix_runtime_code with fix_target='preprocess' to fix both "
                        f"the preprocess code and the schema together."
                    ),
                    "storage": "error",
                    "variable_name": "fix_input_schema_error",
                }]

            # Check if schema actually changed
            if current_schema is not None:
                if json.dumps(new_schema, sort_keys=True) == json.dumps(current_schema, sort_keys=True):
                    return [{
                        "value": (
                            "fix_input_schema: the generated schema is identical to the "
                            "current one. The CONTRACT_ERROR is likely caused by a "
                            "preprocess code issue, not a schema issue. Use fix_runtime_code "
                            "with fix_target='preprocess' instead."
                        ),
                        "storage": "error",
                        "variable_name": "fix_input_schema_error",
                    }]

            # Write via FileTracker
            file_tracker = FileTracker(str(repo_root))
            file_tracker.init_workspace()
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            new_version = file_tracker.modify_file(
                str(schema_path),
                json.dumps(new_schema, ensure_ascii=False, indent=2),
            )
            reason = response.get("reason", "")
            print(
                f"[fix_input_schema] schema updated to FileTracker v{new_version}: "
                f"{[f['field_name'] for f in new_schema.get('inputs', [])]}"
            )

            return [
                {
                    "value": [str(schema_path)],
                    "storage": "temporary",
                    "variable_name": "fix_input_schema_modified_files",
                },
                {
                    "value": {"file_path": str(schema_path), "version": new_version},
                    "storage": "temporary",
                    "variable_name": "fix_input_schema_version",
                },
                {
                    "value": new_schema,
                    "storage": "permanent",
                    "variable_name": "input_schema",
                },
                {
                    "value": reason,
                    "storage": "temporary",
                    "variable_name": "fix_input_schema_summary",
                },
            ]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "fix_input_schema_error",
            }]

    # ──────────────────────────────────────────────────────────────────────
    # Cross-function alignment verification
    # ──────────────────────────────────────────────────────────────────────

    def _assemble_cross_function_alignment_prompt(
        self, service_code: str, fixed_func: str, original_error: str
    ) -> str:
        """Build the post-fix review prompt."""
        p = self.cross_function_alignment_prompt
        prompt_parts = []

        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append("\n".join(p['role']))

        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append("\n".join(p['instructions']))

        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append("\n".join(p['rules']))

        input_text = "\n".join(p['input']).format(
            service_code=service_code,
            fixed_func=fixed_func,
            original_error=original_error,
        )
        prompt_parts.append("\n=== INPUT ===")
        prompt_parts.append(input_text)

        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(json.dumps(p['output_format']))

        return "\n".join(prompt_parts)

    def _verify_cross_function_alignment(
        self, full_service_code: str, fixed_func: str, original_error: str
    ) -> List[str]:
        """Post-fix review: judge whether the targeted fix is sufficient or
        the full service.py needs a complete rewrite.

        Returns a list of concrete issues. Empty list = fix is sufficient.
        """
        prompt = self._assemble_cross_function_alignment_prompt(
            full_service_code, fixed_func, original_error
        )
        try:
            result = json_query(prompt, "cross_function_alignment", self.code_backend)
            if isinstance(result, str):
                result = json.loads(result)
        except Exception as e:
            print(f"[fix_runtime_code] post-fix review failed to run: {e}")
            return []

        needs_full_rewrite = result.get("needs_full_rewrite", False)
        if isinstance(needs_full_rewrite, str):
            needs_full_rewrite = needs_full_rewrite.strip().lower() == "true"
        if not needs_full_rewrite:
            summary = result.get("summary", "")
            print(f"[fix_runtime_code] post-fix review passed: {summary}")
            return []

        issues = result.get("issues", [])
        summary = result.get("summary", "")
        print(f"[fix_runtime_code] post-fix review REJECTED: {summary}")
        return issues if issues else [summary or "post-fix review determined a full rewrite is needed"]

    # ──────────────────────────────────────────────────────────────────────
    # Matcher: test case files -> input_schema fields
    # ──────────────────────────────────────────────────────────────────────
    #
    # Called by validate_api_service / validate_inference_results BEFORE any
    # POST is sent. Turns a case_dir's input/ folder into the exact JSON body
    # that will be sent to /predict_text, using the current input_schema as
    # the spec for what fields the service expects.
    #
    # Flow:
    #   Step 0: manifest override - if case_dir/input/input.json exists, use it directly.
    #   Step 1: read schema from .autodeploy/input_schema.json
    #   Step 2: scan case_dir/input/ and read each file as utf-8 text
    #   Step 3: strict deterministic matching (rule a single/single + rule b exact stem)
    #   Step 4: LLM fallback via json_query("compose_request_body", ...)
    #   Step 5: post-validation (every required field covered, every mapped file exists)
    #   Step 6: build the inputs dict and return
    #
    # Raises RuntimeError (with a clear message) on any failure. Caller is
    # responsible for catching and routing the error to CONTRACT_ERROR.

    _FIELD_NAME_SANITIZE_RE = re.compile(r"[^a-z0-9]+")

    def _load_input_schema(self, repo_root: Path) -> Optional[Dict[str, Any]]:
        schema_path = Path(repo_root) / ".autodeploy" / "input_schema.json"
        if not schema_path.exists():
            return None
        try:
            return json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _scan_case_input_dir(self, case_dir: Path) -> List[Dict[str, Any]]:
        """Return a list of {filename, path, size, content, preview} for every file in case_dir/input/, recursively.

        No filtering by extension (per user decision 4). Every file is read as
        utf-8 text with errors='replace' so binary files produce a noisy
        preview but don't crash the matcher. The caller / LLM can then decide
        to skip them.
        """
        input_dir = Path(case_dir) / "input"
        if not input_dir.is_dir():
            raise RuntimeError(f"case_dir {case_dir} has no input/ directory")

        files = []
        for entry in sorted(input_dir.rglob("*")):
            if not entry.is_file():
                continue
            if entry.name == "input.json" and entry.parent == input_dir:
                # manifest sidecar at top level, handled separately
                continue
            try:
                content = entry.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                content = f"<unreadable: {type(e).__name__}: {e}>"
            files.append({
                "filename": entry.name,
                "path": str(entry.resolve()),
                "size_bytes": entry.stat().st_size,
                "content": content,
                "preview": content[:500],
            })
        return files

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a filename-stem or field-name for exact matching.

        'Passage-Text.TXT'  -> 'passage_text'
        'question '         -> 'question'
        'my_field_123'      -> 'my_field_123'
        """
        lowered = (name or "").strip().lower()
        sanitized = ServiceDelivery._FIELD_NAME_SANITIZE_RE.sub("_", lowered).strip("_")
        return sanitized

    def _try_deterministic_match(
        self,
        schema_fields: List[Dict[str, Any]],
        files: List[Dict[str, Any]],
    ) -> Optional[Dict[str, str]]:
        """Strict deterministic matching. Returns None if not confidently matched.

        Strict mode per user decision 2: only rule a and rule b.
          rule a: exactly 1 required field and exactly 1 file -> match
          rule b: exact normalized filename-stem == normalized field_name -> match
        """
        required = [f for f in schema_fields if f.get("field_required") is True]
        if not required:
            return {}  # nothing required, empty body is fine

        # Rule a: 1 required field + 1 file
        if len(required) == 1 and len(files) == 1:
            return {required[0]["field_name"]: files[0]["path"]}

        # Rule b: exact normalized stem match for every required field
        file_by_norm_stem: Dict[str, str] = {}
        for f in files:
            stem = Path(f["filename"]).stem
            norm = self._normalize_name(stem)
            if norm in file_by_norm_stem:
                # two files with the same normalized stem -> ambiguous,
                # punt to LLM
                return None
            file_by_norm_stem[norm] = f["path"]

        mapping: Dict[str, str] = {}
        for field in required:
            norm_field = self._normalize_name(field["field_name"])
            if norm_field not in file_by_norm_stem:
                return None  # some required field has no exact match -> punt
            mapping[field["field_name"]] = file_by_norm_stem[norm_field]

        # All required fields matched via exact stem
        return mapping

    def _assemble_compose_request_body_prompt(
        self,
        schema: Dict[str, Any],
        files: List[Dict[str, Any]],
    ) -> str:
        prompt_parts = []
        role_prompt = "\n".join(self.compose_request_body_prompt["role"])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append("\n".join(self.compose_request_body_prompt["instructions"]))

        prompt_parts.append("\n=== MATCHING PRIORITY ===")
        prompt_parts.append("\n".join(self.compose_request_body_prompt["matching_priority"]))

        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append("\n".join(self.compose_request_body_prompt["rules"]))

        prompt_parts.append("\n=== INPUT SCHEMA ===")
        prompt_parts.append("\n".join(self.compose_request_body_prompt["input_schema"]).format(
            input_schema_json=json.dumps(schema, ensure_ascii=False, indent=2)
        ))

        # Only send previews to the LLM, not full contents (full content is
        # read after matching succeeds)
        preview_files = [
            {
                "filename": f["filename"],
                "path": f["path"],
                "size_bytes": f["size_bytes"],
                "preview": f["preview"],
            }
            for f in files
        ]
        prompt_parts.append("\n=== AVAILABLE FILES ===")
        prompt_parts.append("\n".join(self.compose_request_body_prompt["available_files"]).format(
            available_files_json=json.dumps(preview_files, ensure_ascii=False, indent=2)
        ))

        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(json.dumps(self.compose_request_body_prompt["output_format"], ensure_ascii=False))

        return "\n".join(prompt_parts)

    def _llm_match(
        self,
        schema: Dict[str, Any],
        files: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Run the LLM matcher. Returns field_name -> absolute path or raises."""
        prompt = self._assemble_compose_request_body_prompt(schema, files)
        response = json_query(prompt, "compose_request_body", self.backend)
        if isinstance(response, str):
            response = json.loads(response)

        mapping_items = response.get("mapping") or []
        reason = response.get("reason", "")

        if not mapping_items:
            raise RuntimeError(
                f"matcher could not confidently map files to schema fields. "
                f"LLM reason: {reason}"
            )

        # Build lookup: accept both path and filename from LLM response
        path_set = {f["path"] for f in files}
        filename_to_path = {}
        for f in files:
            if f["filename"] not in filename_to_path:
                filename_to_path[f["filename"]] = f["path"]
            else:
                filename_to_path[f["filename"]] = None  # ambiguous

        mapping: Dict[str, str] = {}
        for item in mapping_items:
            if not isinstance(item, dict):
                continue
            fname = item.get("field_name")
            file_ref = item.get("path") or item.get("filename")
            if not isinstance(fname, str) or not isinstance(file_ref, str):
                continue
            if file_ref in path_set:
                mapping[fname] = file_ref
            elif file_ref in filename_to_path and filename_to_path[file_ref] is not None:
                mapping[fname] = filename_to_path[file_ref]
            else:
                mapping[fname] = file_ref  # pass through, post-validation will catch
        return mapping

    def _compose_request_body(
        self,
        case_dir: Path,
        repo_root: Path,
    ) -> Dict[str, Any]:
        """Build the `inputs` dict that will be POSTed to /predict_text for a single test case.

        Reads schema, scans files, matches (deterministic then LLM), validates,
        reads file contents, returns a dict like {"passage": "...", "question": "..."}.

        Raises RuntimeError with a human-readable message on any failure. The
        caller in validate_api_service translates that into a CONTRACT_ERROR
        return path.
        """
        case_dir = Path(case_dir)

        # Step 1: load schema
        schema = self._load_input_schema(repo_root)
        if schema is None:
            raise RuntimeError(
                f"cannot compose request body for {case_dir.name}: "
                f"{repo_root}/.autodeploy/input_schema.json missing or unreadable. "
                f"adapt_code must run successfully before validate_api_service."
            )

        schema_inputs = schema.get("inputs") or []
        if not isinstance(schema_inputs, list) or not schema_inputs:
            raise RuntimeError(
                f"input_schema.inputs is empty or not a list; cannot match files to fields."
            )

        # Step 0: manifest override (input/input.json explicit mapping)
        manifest_path = case_dir / "input" / "input.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                return self._compose_from_manifest(manifest, case_dir, schema, repo_root)
            except Exception as e:
                raise RuntimeError(
                    f"input.json manifest in {case_dir.name} could not be parsed: {e}"
                )

        # Step 2: scan files
        files = self._scan_case_input_dir(case_dir)
        if not files:
            raise RuntimeError(
                f"case {case_dir.name}/input/ has no files; cannot match schema fields {[f.get('field_name') for f in schema_inputs]}"
            )

        # Step 3: strict deterministic matching
        mapping = self._try_deterministic_match(schema_inputs, files)
        matcher_used = "deterministic"

        # Step 4: LLM fallback
        if mapping is None:
            print(
                f"[ServiceDelivery] _compose_request_body: deterministic match "
                f"failed for {case_dir.name}, falling back to LLM matcher "
                f"(schema fields = {[f.get('field_name') for f in schema_inputs]}, "
                f"files = {[f['filename'] for f in files]})"
            )
            mapping = self._llm_match(schema, files)
            matcher_used = "llm"

        # Step 5: post-validation
        required_names = {f["field_name"] for f in schema_inputs if f.get("field_required") is True}
        mapped_names = set(mapping.keys())

        missing = required_names - mapped_names
        if missing:
            raise RuntimeError(
                f"matcher ({matcher_used}) did not cover required fields: {sorted(missing)}. "
                f"mapping so far: {mapping}. "
                f"case files: {[f['filename'] for f in files]}"
            )

        declared_names = {f["field_name"] for f in schema_inputs}
        unknown = mapped_names - declared_names
        if unknown:
            raise RuntimeError(
                f"matcher ({matcher_used}) produced mapping for fields not in schema: "
                f"{sorted(unknown)}. declared fields: {sorted(declared_names)}"
            )

        file_by_path = {f["path"]: f for f in files}
        for fname, fpath in mapping.items():
            if fpath not in file_by_path:
                raise RuntimeError(
                    f"matcher ({matcher_used}) mapped field '{fname}' to "
                    f"'{fpath}' which is not in case files. available: "
                    f"{sorted(file_by_path.keys())}"
                )

        # Step 6: copy matched files into repo_root/.autodeploy/inputs/<run_id>/
        # and build the inputs dict with container-internal paths so that the
        # FastAPI app can read the files directly from disk.
        import uuid as _uuid
        run_id = _uuid.uuid4().hex[:12]
        staging_dir = Path(repo_root) / ".autodeploy" / "inputs" / run_id
        staging_dir.mkdir(parents=True, exist_ok=True)

        work_dir = Path(str((global_config.get("docker_setting") or {}).get("work_dir") or "/workspace"))
        container_staging = work_dir / ".autodeploy" / "inputs" / run_id

        inputs: Dict[str, Any] = {}
        for fname, fpath in mapping.items():
            src = Path(fpath)
            dst = staging_dir / src.name
            shutil.copy2(src, dst)
            inputs[fname] = str(container_staging / src.name)

        print(
            f"[ServiceDelivery] _compose_request_body: {case_dir.name} matched via "
            f"{matcher_used}, files staged to {staging_dir}: "
            f"{dict((k, file_by_path[mapping[k]]['filename']) for k in mapping)}"
        )
        return inputs

    def _compose_from_manifest(
        self,
        manifest: Dict[str, Any],
        case_dir: Path,
        schema: Dict[str, Any],
        repo_root: Path,
    ) -> Dict[str, Any]:
        """Explicit manifest-driven body composition.

        Manifest format (input/input.json):
            {"fields": {"passage":  {"kind": "text_file", "path": "p.txt"},
                        "question": {"kind": "inline",    "value": "What...?"}}}

        Used to let users override the matcher for weird test cases.
        For text_file entries the file is copied into
        repo_root/.autodeploy/inputs/<run_id>/ and the container-internal
        path is placed in the inputs dict.
        """
        import uuid as _uuid

        fields = manifest.get("fields") or {}
        if not isinstance(fields, dict):
            raise RuntimeError(f"input.json manifest 'fields' must be a dict, got {type(fields).__name__}")

        run_id = _uuid.uuid4().hex[:12]
        staging_dir = Path(repo_root) / ".autodeploy" / "inputs" / run_id
        staging_dir.mkdir(parents=True, exist_ok=True)

        work_dir = Path(str((global_config.get("docker_setting") or {}).get("work_dir") or "/workspace"))
        container_staging = work_dir / ".autodeploy" / "inputs" / run_id

        inputs: Dict[str, Any] = {}
        for fname, spec in fields.items():
            if not isinstance(spec, dict):
                continue
            kind = spec.get("kind")
            if kind == "inline":
                inputs[fname] = spec.get("value", "")
            elif kind == "text_file":
                path = spec.get("path", "")
                file_path = case_dir / "input" / path
                if not file_path.exists():
                    raise RuntimeError(
                        f"manifest field '{fname}' points to {path} which does not exist under {case_dir}/input/"
                    )
                dst = staging_dir / Path(path).name
                shutil.copy2(file_path, dst)
                inputs[fname] = str(container_staging / Path(path).name)
            else:
                raise RuntimeError(
                    f"manifest field '{fname}' has unsupported kind {kind!r} "
                    f"(expected 'inline' or 'text_file')"
                )

        # Validate against schema
        required_names = {
            f["field_name"] for f in (schema.get("inputs") or [])
            if f.get("field_required") is True
        }
        missing = required_names - set(inputs.keys())
        if missing:
            raise RuntimeError(
                f"manifest does not provide required schema fields: {sorted(missing)}"
            )
        return inputs

    def _get_runtime_ports(self, container_id: Optional[str] = None) -> tuple[int, int]:
        """Return (host_port, container_port) for the running service.

        Source priority (stop at first success):
          1. `docker port <container_id>` - real truth from the daemon, works
             even when DockerSetUp.docker_run auto-picked a fallback host port
             after a collision. Requires container_id.
          2. `docker_setting.port_mapping` config - static fallback.
          3. Hard default 8000:8000.
        """
        if container_id:
            try:
                res = subprocess.run(
                    ["docker", "port", str(container_id)],
                    capture_output=True, text=True, timeout=5,
                )
                if res.returncode == 0 and res.stdout.strip():
                    # Output lines look like: "8000/tcp -> 0.0.0.0:18000"
                    for line in res.stdout.splitlines():
                        line = line.strip()
                        if "->" not in line:
                            continue
                        left, right = [s.strip() for s in line.split("->", 1)]
                        ctr_part = left.split("/", 1)[0]
                        host_part = right.rsplit(":", 1)[-1]
                        if ctr_part.isdigit() and host_part.isdigit():
                            return int(host_part), int(ctr_part)
            except Exception:
                pass

        port_mapping = str(docker_setting.get("port_mapping") or "8000:8000").strip()
        parts = [part.strip() for part in port_mapping.split(":") if part.strip()]
        if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
            return int(parts[-2]), int(parts[-1])
        if len(parts) == 1 and parts[0].isdigit():
            port = int(parts[0])
            return port, port
        return 8000, 8000

    def _to_container_repo_path(self, repo_root: Path, host_path: Path) -> Path:
        repo_root = Path(repo_root).resolve()
        host_path = Path(host_path).resolve()
        work_dir = Path(str(docker_setting.get("work_dir") or "/workspace"))
        relative_path = host_path.relative_to(repo_root)
        return work_dir / relative_path

    def _wait_for_service_ready(
        self,
        host_port: int,
        timeout_seconds: int = 600,
        interval_seconds: float = 3.0,
        container_id: str = "",
        api_pid: str = "",
    ) -> None:
        """Wait for the service health-check to return 200.

        If *container_id* and *api_pid* are provided, the wait loop also
        checks whether the uvicorn process is still alive inside the
        container. This avoids waiting the full timeout when the process
        has already crashed — we fail fast with a clear message that
        includes the last few lines of the container log.

        When the process IS still alive (i.e. the model is still loading),
        the loop keeps waiting up to *timeout_seconds* (default 600 s /
        10 minutes) so large models have time to load.
        """
        deadline = time.time() + timeout_seconds
        last_error = "service did not become ready"
        checks_since_last_alive = 0

        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{host_port}/", timeout=3) as response:
                    if response.status == 200:
                        return
            except Exception as e:
                last_error = str(e)

            # If we know the PID, check whether the process is still alive
            # every few iterations. If it died, fail immediately instead of
            # waiting out the full timeout.
            checks_since_last_alive += 1
            if container_id and api_pid and checks_since_last_alive >= 3:
                checks_since_last_alive = 0
                try:
                    alive_check = linux_command_in_docker(
                        f"kill -0 {api_pid} 2>/dev/null && echo alive || echo dead",
                        container_id,
                    )
                    if "dead" in alive_check.stdout:
                        raise RuntimeError(
                            f"uvicorn process {api_pid} exited during model loading "
                            f"(health check never succeeded). last error: {last_error}"
                        )
                except RuntimeError:
                    raise
                except Exception:
                    pass  # docker exec itself failed; keep waiting

            time.sleep(interval_seconds)
        raise RuntimeError(
            f"Service did not become ready within {timeout_seconds}s. "
            f"last error: {last_error}"
        )

    def _post_inputs_json(self, inputs: Dict[str, Any], host_port: int, case_label: str) -> Dict[str, Any]:
        """POST `inputs` as a JSON body to /predict_text and return the parsed response.

        Replaces the old hand-written multipart /predict call. The request
        body shape is {"inputs": {<field>: <value>, ...}} - no `params` key
        (that was removed from the schema entirely). FastAPI validates
        against .autodeploy/input_schema.json on its side and, if valid,
        calls predict(raw_input=inputs, model_bundle=...).

        Raises RuntimeError on non-200, non-success payload, or network
        error. The caller converts that into a CONTRACT_ERROR / RUNTIME_ERROR
        depending on the message.
        """
        body = json.dumps({"inputs": inputs}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{host_port}/predict_text",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            response_body = response.read().decode("utf-8", errors="ignore")
            payload = json.loads(response_body)
            if response.status != 200:
                raise RuntimeError(f"Unexpected HTTP status {response.status} for case {case_label}")
            if not payload.get("success"):
                raise RuntimeError(f"Inference failed for case {case_label}: {payload.get('error', 'unknown error')}")
            return {
                "case_name": case_label,
                "status_code": response.status,
                "success": True,
                "response_payload": payload,
            }

    def _select_smoke_test_case(self, test_file_dir: Path) -> Path:
        """Pick the first case_dir that has an input/ directory with at least one file (recursive).

        Replaces _select_smoke_test_input: no longer returns a specific file,
        just the case directory. The matcher inside _compose_request_body
        will pick which file(s) to use based on the schema.
        """
        case_dirs = [path for path in sorted(test_file_dir.iterdir()) if path.is_dir()]
        for case_dir in case_dirs:
            input_dir = case_dir / "input"
            if not input_dir.is_dir():
                continue
            # Recursively find any file (excluding top-level input.json manifest)
            input_files = [
                p for p in input_dir.rglob("*")
                if p.is_file() and not (p.name == "input.json" and p.parent == input_dir)
            ]
            if input_files:
                return case_dir
            # Even if there are no raw files but there's an input.json manifest,
            # the manifest might provide inline values - treat that as valid too.
            if (input_dir / "input.json").exists():
                return case_dir
        raise RuntimeError(
            f"No smoke test case found in {test_file_dir}. "
            "Expected test_file_dir/<case_name>/input/<file(s) or input.json>."
        )

    def _list_case_dirs(self, test_file_dir: Path) -> list[Path]:
        return [path for path in sorted(test_file_dir.iterdir()) if path.is_dir() and (path / "input").is_dir()]

    def _run_case_inference_via_api(self, case_dir: Path, host_port: int, repo_root: Path) -> Dict[str, Any]:
        inputs = self._compose_request_body(case_dir, repo_root)
        api_result = self._post_inputs_json(inputs, host_port, case_dir.name)
        return {
            "case_name": case_dir.name,
            "request_inputs": inputs,
            "api_result": api_result,
        }

    def _read_output_format_guide(self, test_file_dir: Path) -> str:
        guide_path = test_file_dir / "OUTPUT_FORMAT.md"
        if not guide_path.is_file():
            return ""
        return guide_path.read_text(encoding="utf-8", errors="ignore").strip()

    def _load_case_expected_output(self, case_dir: Path) -> str:
        output_dir = case_dir / "output"
        if not output_dir.is_dir():
            raise RuntimeError(f"Missing output directory in {case_dir}")

        parts: List[str] = []
        for output_file in sorted(output_dir.iterdir()):
            if not output_file.is_file():
                continue
            parts.append(output_file.read_text(encoding="utf-8", errors="ignore").strip())

        if not parts:
            raise RuntimeError(f"No expected output file found in {output_dir}")
        return "\n\n".join(parts)

    def _assemble_llm_normalize_prompt(self, output_format_reference: str, actual_result: str, api_response_result: str) -> str:
        output_format_reference_text = output_format_reference or "No OUTPUT_FORMAT.md found."
        prompt_parts = []

        role_prompt = "\n".join(self.llm_normalize_prompt["role"])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        instructions_prompt = "\n".join(self.llm_normalize_prompt["instructions"])
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append(instructions_prompt)

        judge_types_prompt = "\n".join(self.llm_normalize_prompt["judge_types"])
        prompt_parts.append("\n=== JUDGE TYPES ===")
        prompt_parts.append(judge_types_prompt)

        output_format_reference_prompt = "\n".join(self.llm_normalize_prompt["output_format_reference"]).format(
            output_format_reference=output_format_reference_text
        )
        prompt_parts.append("\n=== OUTPUT_FORMAT.MD REFERENCE ===")
        prompt_parts.append(output_format_reference_prompt)

        actual_result_prompt = "\n".join(self.llm_normalize_prompt["actual_result"]).format(
            actual_result=actual_result
        )
        prompt_parts.append("\n=== ACTUAL RESULT ===")
        prompt_parts.append(actual_result_prompt)

        api_response_result_prompt = "\n".join(self.llm_normalize_prompt["api_response_result"]).format(
            api_response_result=api_response_result
        )
        prompt_parts.append("\n=== API RESPONSE RESULT ===")
        prompt_parts.append(api_response_result_prompt)

        rules_prompt = "\n".join(self.llm_normalize_prompt["rules"])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)

        output_format_prompt = json.dumps(self.llm_normalize_prompt["output_format"], ensure_ascii=False)
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        return "\n".join(prompt_parts)

    def _normalize_list_strings(self, values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        normalized = []
        for value in values:
            text = self._normalize_text(value)
            if text:
                normalized.append(text)
        return sorted(set(normalized))

    def _compare_field_exact(self, expected: Any, actual: Any, min_similarity: float = 0.5) -> Dict[str, Any]:
        expected_text = self._normalize_text(expected)
        actual_text = self._normalize_text(actual)
        score = difflib.SequenceMatcher(None, expected_text, actual_text).ratio()
        passed = score >= min_similarity
        return {
            "passed": passed,
            "score": score,
        }

    def _compare_field_set_match(self, expected: Any, actual: Any, min_f1: float = 0.6) -> Dict[str, Any]:
        expected_items = set(self._normalize_list_strings(expected))
        actual_items = set(self._normalize_list_strings(actual))
        intersection = expected_items & actual_items
        precision = len(intersection) / len(actual_items) if actual_items else (1.0 if not expected_items else 0.0)
        recall = len(intersection) / len(expected_items) if expected_items else 1.0
        score = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        passed = score >= min_f1
        return {
            "passed": passed,
            "score": score,
        }

    def _compare_field_numeric_tolerance(self, expected: Any, actual: Any, max_pct: float = 0.10) -> Dict[str, Any]:
        try:
            expected_value = float(expected)
            actual_value = float(actual)
        except Exception as exc:
            raise RuntimeError(f"numeric_tolerance expects numeric scalar values: {exc}")
        if expected_value == 0:
            pct_diff = 0.0 if actual_value == 0 else 1.0
        else:
            pct_diff = abs(actual_value - expected_value) / abs(expected_value)
        passed = pct_diff <= max_pct
        score = max(0.0, 1.0 - pct_diff)
        return {
            "passed": passed,
            "score": score,
        }

    def _evaluate_llm_judge_result(self, judge_result: Dict[str, Any]) -> Dict[str, Any]:
        normalized_expected_list = judge_result.get("normalized_expected") or []
        normalized_actual_list = judge_result.get("normalized_actual") or []
        field_judges_list = judge_result.get("field_judges") or []
        if not isinstance(normalized_expected_list, list) or not isinstance(normalized_actual_list, list):
            raise RuntimeError("normalized_expected and normalized_actual must be lists")
        if not isinstance(field_judges_list, list) or not field_judges_list:
            return {
                "passed": False,
                "fields": {},
                "no_comparable_fields": True,
                "reason": (
                    "The normalizer could not extract any comparable fields between "
                    "expected output and API response. This usually means the API "
                    "response format is too different from the expected output. "
                    "Use fix_runtime_code with fix_target='postprocess' to adapt "
                    "the output format, or fix_target='all' if the inference logic "
                    "also needs adjustment."
                ),
            }

        # Output format alignment check: as long as the normalizer could
        # extract comparable fields from both sides, the format is consistent
        # and validation passes. We still compute scores for diagnostics but
        # they don't affect the pass/fail decision.
        normalized_expected = {item["field_name"]: item["field_value"] for item in normalized_expected_list}
        normalized_actual = {item["field_name"]: item["field_value"] for item in normalized_actual_list}

        allowed_types = {"exact", "set_match", "numeric_tolerance"}
        field_results: Dict[str, Any] = {}

        for judge in field_judges_list:
            field_name = judge["field_name"]
            field_judge_type = str(judge.get("field_judge_type") or "").strip().lower()
            if field_judge_type not in allowed_types:
                field_judge_type = "exact"

            if field_name not in normalized_expected or field_name not in normalized_actual:
                field_results[field_name] = {"score": 0.0}
                continue

            if field_judge_type == "exact":
                field_result = self._compare_field_exact(normalized_expected[field_name], normalized_actual[field_name])
            elif field_judge_type == "set_match":
                field_result = self._compare_field_set_match(normalized_expected[field_name], normalized_actual[field_name])
            else:
                field_result = self._compare_field_numeric_tolerance(normalized_expected[field_name], normalized_actual[field_name])

            field_results[field_name] = {"score": field_result["score"]}

        return {
            "passed": True,
            "fields": field_results,
        }

    def _normalize_with_llm(
        self,
        output_format_reference: str,
        actual_result: str,
        api_response_result: str,
    ) -> Dict[str, Any]:
        prompt = self._assemble_llm_normalize_prompt(
            output_format_reference=output_format_reference,
            actual_result=actual_result,
            api_response_result=api_response_result,
        )
        response = json_query(prompt, "llm_normalize", self.backend)
        if not isinstance(response, dict):
            raise RuntimeError("LLM normalization response must be a JSON object")
        return response

    def _run_field_judges(self, normalization: Dict[str, Any]) -> Dict[str, Any]:
        return self._evaluate_llm_judge_result(normalization)

    def _normalize_text(self, value: Any) -> str:
        return " ".join(str(value).strip().lower().split())

    def validate_api_service(self, repo_root: Path, container_id: str, test_file_dir: Path, fastapiapp_dir: Path):
        api_pid = ""
        container_port = 8000  # safe default; overwritten by _get_runtime_ports
        repo_root = Path(repo_root).resolve()
        test_file_dir = Path(test_file_dir)
        fastapiapp_dir = Path(fastapiapp_dir)
        host_api_log_path = repo_root / ".autodeploy" / "uvicorn.log"
        container_api_log_path = self._to_container_repo_path(repo_root, host_api_log_path)
        container_fastapiapp_dir = self._to_container_repo_path(repo_root, fastapiapp_dir)
        try:
            host_port, container_port = self._get_runtime_ports(container_id)

            # Acquire GPU lock before starting uvicorn (model loading uses GPU)
            # and running the smoke test (inference uses GPU). The lock is held
            # until we finish or fail, so other containers wait their turn.
            with gpu_lock():
                # Release any process still holding the container port before
                # starting uvicorn. Without this, a previous uvicorn that was
                # killed but hasn't fully exited yet causes the new instance to
                # log "address already in use" and exit immediately, overwriting
                # the real error (IMPORT_ERROR / MODEL_LOAD_ERROR) in uvicorn.log
                # with a misleading PORT_ERROR that sends the agent on the wrong
                # fix path.
                # Kill any old uvicorn process so the port is free.
                # NOTE: use the [u]vicorn bracket trick so that pkill/pgrep
                # do NOT match the bash -c wrapper spawned by docker exec
                # (whose cmdline also contains "uvicorn").  Without this,
                # pkill sends SIGKILL to the wrapper bash, docker exec
                # returns non-zero, and the real uvicorn survives.
                linux_command_in_docker(
                    "pkill -9 -f '[u]vicorn' 2>/dev/null || true", container_id,
                )
                # Poll until uvicorn is fully dead AND port is released (up to 10s).
                for _attempt in range(10):
                    time.sleep(1)
                    pgrep_check = linux_command_in_docker(
                        "pgrep -f '[u]vicorn' 2>/dev/null || true", container_id,
                    )
                    port_check = linux_command_in_docker(
                        f"fuser {container_port}/tcp 2>/dev/null || true", container_id,
                    )
                    if not pgrep_check.stdout.strip() and not port_check.stdout.strip():
                        break
                else:
                    # Last resort: kill again and force-free the port.
                    linux_command_in_docker(
                        "pkill -9 -f '[u]vicorn' 2>/dev/null || true", container_id,
                    )
                    linux_command_in_docker(
                        f"fuser -k {container_port}/tcp 2>/dev/null || true", container_id,
                    )
                    time.sleep(2)

                start_api_command = (
                    f'mkdir -p "{container_api_log_path.parent}" && '
                    f'echo "--- new attempt $(date) ---" >> "{container_api_log_path}" && '
                    f'nohup python3 -m uvicorn app:app --app-dir "{container_fastapiapp_dir}" '
                    f'--host 0.0.0.0 --port {container_port} >> "{container_api_log_path}" 2>&1 & echo $!'
                )
                api_pid_res = linux_command_in_docker(start_api_command, container_id)
                api_pid = api_pid_res.stdout.strip()

                self._wait_for_service_ready(
                    host_port,
                    container_id=container_id,
                    api_pid=api_pid,
                )

                # Verify the process that answered the health check is the
                # one we just started, not a leftover from a previous run.
                pid_check = linux_command_in_docker(
                    f"kill -0 {api_pid} 2>/dev/null && echo alive || echo dead",
                    container_id,
                )
                if "dead" in pid_check.stdout:
                    raise RuntimeError(
                        f"uvicorn process {api_pid} exited during startup. "
                        f"The health check may have hit a stale process."
                    )

                # Pick one test case for the smoke test and use the matcher to
                # compose a valid request body that satisfies the deployed
                # service's input_schema. Post it as JSON to /predict_text.
                smoke_test_case_dir = self._select_smoke_test_case(test_file_dir)
                smoke_inputs = self._compose_request_body(smoke_test_case_dir, repo_root)
                smoke_test_result = self._post_inputs_json(
                    smoke_inputs, host_port, smoke_test_case_dir.name
                )
                smoke_test_result["request_inputs"] = smoke_inputs

            return [{
                "value": api_pid,
                "storage": "permanent",
                "variable_name": "api_pid"
            },
            {
                "value": smoke_test_result,
                "storage": "temporary",
                "variable_name": "smoke_test_result"
            },
            {
                "value": str(host_api_log_path),
                "storage": "permanent",
                "variable_name": "api_log_path"
            },
            {
                "value": "Successfully validated the API service.",
                "storage": "permanent",
                "variable_name": "service_delivery_success_flag"
            }]
        except Exception as e:
            # Read the container's uvicorn log FIRST so the analyzer sees the
            # real traceback (where torch / CUDA / import errors live), not just
            # the high-level "connection refused" thrown by _wait_for_service_ready.
            # Without this, CUDA/torch mismatches get mis-classified as
            # TIMEOUT_ERROR / PORT_ERROR / PROCESS_ERROR and routed to
            # fix_runtime_code, which can't repair Dockerfile-level issues.
            try:
                api_log_res = linux_command_in_docker(f'cat "{container_api_log_path}"', container_id)
                api_log_content = api_log_res.stdout
            except Exception:
                api_log_content = ""

            # Only keep the LAST attempt block so the error analyzer
            # doesn't get confused by stale errors from earlier attempts.
            _ATTEMPT_SEPARATOR = "--- new attempt "
            if _ATTEMPT_SEPARATOR in api_log_content:
                last_block_idx = api_log_content.rfind(_ATTEMPT_SEPARATOR)
                api_log_content = api_log_content[last_block_idx:]

            if len(api_log_content) > 4000:
                log_excerpt = (
                    api_log_content[:2000]
                    + "\n...[truncated middle]...\n"
                    + api_log_content[-2000:]
                )
            else:
                log_excerpt = api_log_content
            error_text = (
                f"--- container api log ---\n{log_excerpt}\n\n"
                f"--- exception raised by health check ---\n{str(e)}"
            )
            analyze_error_prompt = self.assemble_analyze_error_prompt(error_text)
            analyze_error = json_query(analyze_error_prompt, "analyze_error", self.backend)
            if isinstance(analyze_error, str):
                analyze_error = json.loads(analyze_error)
            error_classification = analyze_error['error_classification']
            error_suggestion = analyze_error['error_suggestion']

            # Kill the new uvicorn we just started AND any stale process
            # still holding the port.  Previously only api_pid (the new
            # process, often already dead) was killed, leaving the old
            # uvicorn alive and perpetuating the PORT_ERROR loop.
            if api_pid:
                try:
                    linux_command_in_docker(f"kill -9 {api_pid} 2>/dev/null || true", container_id)
                except Exception:
                    pass
            try:
                linux_command_in_docker("pkill -9 -f '[u]vicorn' 2>/dev/null || true", container_id)
            except Exception:
                pass
            return [{
                "value": error_classification,
                "storage": "temporary",
                "variable_name": "service_delivery_error_class"
            },
            {
                "value": error_suggestion,
                "storage": "temporary",
                "variable_name": "service_delivery_error_suggestion"
            },
            {
                "value": str(host_api_log_path),
                "storage": "permanent",
                "variable_name": "api_log_path"
            },
            {
                "value": "Failed to run the model, use a tool to fix the error",
                "storage": "error",
                "variable_name": "service_delivery_error_flag"
            }]

    def validate_inference_results(self, repo_root: Path, container_id: str, api_pid: str, test_file_dir: Path):
        try:
            if not str(container_id).strip():
                raise RuntimeError("validate_inference_results requires a running Docker container. Missing container_id.")
            if not str(api_pid).strip():
                raise RuntimeError("validate_inference_results requires a running API process. Missing api_pid.")
            repo_root = Path(repo_root).resolve()
            test_file_dir = Path(test_file_dir)
            host_port, _ = self._get_runtime_ports(container_id)
            # Use inspect_running (single source of truth) instead of `docker ps`.
            # `docker ps` counts `restarting` containers as visible, which would
            # let a crash-looping container sneak past this guard and then fail
            # on the next docker exec with "Container is restarting".
            if not inspect_running(container_id):
                raise RuntimeError(f"Container {container_id} is not running.")
            self._wait_for_service_ready(
                host_port,
                container_id=container_id,
                api_pid=api_pid,
            )
            output_format_reference = self._read_output_format_guide(test_file_dir)
            case_results = []

            with gpu_lock():
                for case_dir in self._list_case_dirs(test_file_dir):
                    inference_run = self._run_case_inference_via_api(case_dir, host_port, repo_root)
                    expected_output = self._load_case_expected_output(case_dir)
                    api_result = inference_run["api_result"]
                    api_response_result = json.dumps(api_result["response_payload"], ensure_ascii=False)

                    # Save inference output for this case
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    inference_output_dir = case_dir / f"inference_output_{timestamp}"
                    inference_output_dir.mkdir(parents=True, exist_ok=True)
                    (inference_output_dir / "response_payload.json").write_text(
                        api_response_result, encoding="utf-8"
                    )

                    # Heuristic: detect suspiciously short output that suggests
                    # a generative model only ran a single forward pass instead
                    # of model.generate().  Compare the actual result length to
                    # the expected output length — if the actual is dramatically
                    # shorter, flag it so the agent can target the root cause.
                    truncation_warning = None
                    expected_len = len(expected_output)
                    actual_len = len(api_response_result)
                    if expected_len > 20 and actual_len < expected_len * 0.1:
                        truncation_warning = (
                            f"POSSIBLE TRUNCATED GENERATION: expected output is ~{expected_len} chars "
                            f"but API returned only ~{actual_len} chars. This often means the inference() "
                            f"function only does a single model() / model.forward() call instead of "
                            f"producing the full output sequence. Fix: replicate the generation method "
                            f"from the original code (model.generate(), manual autoregressive loop, "
                            f"or whatever the original repo uses) with appropriate generation parameters."
                        )
                        print(f"[ServiceDelivery] WARNING: {truncation_warning}")

                    normalization = self._normalize_with_llm(
                        output_format_reference=output_format_reference,
                        actual_result=expected_output,
                        api_response_result=api_response_result,
                    )
                    comparison = self._run_field_judges(normalization)
                    if comparison.get("no_comparable_fields"):
                        comparison["reason"] = (
                            f"{comparison['reason']}\n"
                            f"Expected output (preview): {expected_output[:500]}\n"
                            f"Actual API response (preview): {api_response_result[:500]}"
                        )
                    case_entry = {
                        "case_name": inference_run["case_name"],
                        "request_inputs": inference_run["request_inputs"],
                        "expected_output": expected_output,
                        "api_response_result": api_response_result,
                        **comparison,
                    }
                    if truncation_warning:
                        case_entry["truncation_warning"] = truncation_warning
                    case_results.append(case_entry)

            if not case_results:
                raise RuntimeError(
                    f"No validation cases found in {test_file_dir}. "
                    "Expected test_file_dir/<case_name>/input and test_file_dir/<case_name>/output."
                )

            passed_cases = [item for item in case_results if item["passed"]]
            failed_cases = [item for item in case_results if not item["passed"]]

            observations = [{
                "value": case_results,
                "storage": "temporary",
                "variable_name": "validation_results"
            },
            {
                "value": failed_cases,
                "storage": "temporary",
                "variable_name": "validation_failures"
            }]
            # At least one case passed → treat inference as successful.
            if not passed_cases:
                observations.append({
                    "value": (
                        f"All {len(case_results)} test cases failed inference "
                        f"correctness check. Call fix_runtime_code with "
                        f"service_delivery_error_class='INFERENCE_LOGIC_ERROR', "
                        f"service_delivery_error_suggestion synthesized from validation_failures, "
                        f"and service_delivery_api_log read from api_log_path. "
                        f"Then call stop_api_service and re-run validate_api_service."
                    ),
                    "storage": "error",
                    "variable_name": "validation_correctness_error"
                })
            return observations
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "validation_error"
            }]

    # ------------------------------------------------------------------
    # final delivery + deployment finalize (formerly EndPhase)
    # ------------------------------------------------------------------

    def _build_api_doc_base_info(self, repo_root: Path, container_id: str, api_pid: str, test_file_dir: Optional[Path] = None) -> str:
        """Build deterministic base info block (runtime, schema, example) for the API doc prompt."""
        parts = []

        # Runtime info
        host_port, container_port = self._get_runtime_ports(container_id)
        parts.append("=== RUNTIME INFO ===")
        parts.append(f"Container ID: {container_id}")
        parts.append(f"API PID: {api_pid}")
        parts.append(f"Host Port: {host_port}")
        parts.append(f"Container Port: {container_port}")
        parts.append(f"Base URL: http://localhost:{host_port}")

        # Input schema
        schema_path = Path(repo_root).resolve() / ".autodeploy" / "input_schema.json"
        if schema_path.exists():
            try:
                schema_text = schema_path.read_text(encoding="utf-8")
                parts.append("\n=== INPUT SCHEMA (.autodeploy/input_schema.json) ===")
                parts.append(schema_text)
            except Exception:
                pass

        # Verified example (request body + response)
        if test_file_dir is not None:
            try:
                for case_dir in self._list_case_dirs(Path(test_file_dir)):
                    try:
                        inference_dirs = sorted(
                            [d for d in case_dir.iterdir() if d.is_dir() and d.name.startswith("inference_output_")],
                            reverse=True,
                        )
                        if not inference_dirs:
                            continue
                        example_parts = []
                        req_file = inference_dirs[0] / "request_body.json"
                        if req_file.is_file():
                            example_parts.append("Request body:")
                            example_parts.append(req_file.read_text(encoding="utf-8", errors="replace")[:2000])
                        resp_file = inference_dirs[0] / "response_payload.json"
                        if resp_file.is_file():
                            example_parts.append("Response:")
                            example_parts.append(resp_file.read_text(encoding="utf-8", errors="replace")[:2000])
                        if example_parts:
                            parts.append(f"\n=== VERIFIED EXAMPLE (case: {case_dir.name}) ===")
                            parts.extend(example_parts)
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        return "\n".join(parts)

    def assemble_api_doc_prompt(self, task: str, service_pipeline_path: Path, fastapiapp_dir: Path, cleaned_readmes_path: Path, base_info: str, test_file_dir: Optional[Path] = None) -> str:
        prompt_parts = []

        role_prompt = "\n".join(self.api_doc_prompt["role"])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        instructions_prompt = "\n".join(self.api_doc_prompt["instructions"])
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append(instructions_prompt)

        rules_prompt = "\n".join(self.api_doc_prompt["rules"])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)

        task_prompt = "\n".join(self.api_doc_prompt["task"]).format(task=task)
        prompt_parts.append("\n=== TASK ===")
        prompt_parts.append(task_prompt)

        with open(cleaned_readmes_path, "r", encoding="utf-8", errors="ignore") as f:
            cleaned_readme = f.read()
        readme_prompt = "\n".join(self.api_doc_prompt["readme_content"]).format(
            readme_content=cleaned_readme
        )
        prompt_parts.append("\n=== README CONTENT ===")
        prompt_parts.append(readme_prompt)

        with open(service_pipeline_path, "r", encoding="utf-8", errors="ignore") as f:
            service_code = f.read()
        service_prompt = "\n".join(self.api_doc_prompt["service_code"]).format(
            service_pipeline_path=service_pipeline_path,
            service_code=service_code,
        )
        prompt_parts.append("\n=== SERVICE CODE ===")
        prompt_parts.append(service_prompt)

        app_path = Path(fastapiapp_dir) / "app.py"
        with open(app_path, "r", encoding="utf-8", errors="ignore") as f:
            app_code = f.read()
        app_prompt = "\n".join(self.api_doc_prompt["fastapi_app_code"]).format(
            app_path=app_path,
            app_code=app_code,
        )
        prompt_parts.append("\n=== FASTAPI APP CODE ===")
        prompt_parts.append(app_prompt)

        # Deterministic base info (runtime, schema, example)
        prompt_parts.append("\n" + base_info)

        output_format_prompt = "\n".join(self.api_doc_prompt["output_format"])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        return "\n".join(prompt_parts)

    def generate_api_documentation(self, repo_root: Path, task: str, service_pipeline_path: Path, fastapiapp_dir: Path, cleaned_readmes_path: Path, test_file_dir: Path, container_id: str, api_pid: str):
        try:
            repo_root = Path(repo_root).resolve()

            base_info = self._build_api_doc_base_info(
                repo_root=repo_root,
                container_id=container_id,
                api_pid=api_pid,
                test_file_dir=Path(test_file_dir) if test_file_dir else None,
            )

            prompt = self.assemble_api_doc_prompt(
                task=task,
                service_pipeline_path=service_pipeline_path,
                fastapiapp_dir=fastapiapp_dir,
                cleaned_readmes_path=cleaned_readmes_path,
                base_info=base_info,
                test_file_dir=Path(test_file_dir) if test_file_dir else None,
            )
            api_doc_markdown = query(prompt, self.backend)

            output_dir = repo_root / ".autodeploy"
            output_dir.mkdir(parents=True, exist_ok=True)
            api_doc_path = output_dir / "API_DOCUMENTATION.md"
            api_doc_path.write_text(api_doc_markdown, encoding="utf-8")

            return [{
                "value": str(api_doc_path),
                "storage": "permanent",
                "variable_name": "api_documentation_path"
            },
            {
                "value": "API documentation generated successfully.",
                "storage": "temporary",
                "variable_name": "api_documentation_generated"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "api_documentation_error"
            }]

    def end_deployment(self, repo_root: Path, api_pid: str, api_log_path: Path, api_documentation_path: Path):
        return [{
            "value": str(repo_root),
            "storage": "permanent",
            "variable_name": "repo_root"
        },
        {
            "value": api_pid,
            "storage": "permanent",
            "variable_name": "api_pid"
        },
        {
            "value": api_log_path,
            "storage": "permanent",
            "variable_name": "api_log_path"
        },
        {
            "value": api_documentation_path,
            "storage": "permanent",
            "variable_name": "api_documentation_path"
        }]
