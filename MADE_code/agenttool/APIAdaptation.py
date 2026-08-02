import json
import os
import re
import traceback
from abc import ABC, abstractmethod
import yaml
import ast
import shutil
from pathlib import Path
from typing import Union, List, Dict, Callable
from backend.query import query, code_query, json_query
from agenttool.base_phase import BasePhase
from agenttool.tool import *
from agenttool.entry_find_tool import EntryFindTool
from agenttool.FileTracker import FileTracker

file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)

try:
    with open(os.path.join(project_path, "config/global.yaml"), "r") as f:
        global_config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Config file not found.")
except yaml.YAMLError as exc:
    raise yaml.YAMLError(f"Error in configuration file: {exc}")


# ───────────────────────────────────────────────────────────────────────────
# input_schema structural + consistency helpers
# ───────────────────────────────────────────────────────────────────────────
#
# Used by adapt_code (preprocess stage) and by ServiceDelivery.fix_runtime_code
# to keep service.py and .autodeploy/input_schema.json atomically in sync.
# All checks are deterministic and zero-LLM - they are the post-validation
# layer that catches "LLM declared fields different from the fields its code
# actually reads" drift before anything is written to disk.


_INPUT_SCHEMA_ALLOWED_TYPES = {"string", "number", "integer", "boolean"}
_INPUT_SCHEMA_FIELD_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

# Regex that captures every `raw_input['<field>']` / `raw_input["<field>"]`
# access in preprocess source code. Used by _check_preprocess_schema_consistency
# to verify the generated preprocess actually reads the fields its schema claims.
_RAW_INPUT_KEY_RE = re.compile(r"raw_input\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]")


def _validate_input_schema_structure(schema):
    """Return a list of structural errors. Empty list == schema is well-formed.

    Expected shape (list form, matches input_schema_schema in agent/output_store.py):
        {
          "version": "1.0",
          "task_type": "<label>",
          "inputs": [
            {"field_name": "...", "field_type": "string", "field_required": true, "field_description": "..."},
            ...
          ],
          "reason": "..."
        }

    There is no `params` key anymore (removed per design decision).
    """
    errors = []
    if not isinstance(schema, dict):
        return [f"input_schema must be a dict, got {type(schema).__name__}"]

    inputs = schema.get("inputs")
    if not isinstance(inputs, list):
        return [
            f"input_schema.inputs must be a list, got {type(inputs).__name__}. "
            "Use list form: [{field_name, field_type, field_required, field_description}, ...]"
        ]
    if not inputs:
        errors.append("input_schema.inputs must contain at least one field")

    seen_names = set()
    for idx, item in enumerate(inputs):
        if not isinstance(item, dict):
            errors.append(f"input_schema.inputs[{idx}] must be a dict, got {type(item).__name__}")
            continue

        name = item.get("field_name")
        if not isinstance(name, str) or not _INPUT_SCHEMA_FIELD_NAME_RE.match(name or ""):
            errors.append(
                f"input_schema.inputs[{idx}].field_name {name!r} must match [a-z_][a-z0-9_]*"
            )
            continue
        if name in seen_names:
            errors.append(f"input_schema.inputs has duplicate field_name '{name}'")
            continue
        seen_names.add(name)

        ftype = item.get("field_type")
        if ftype not in _INPUT_SCHEMA_ALLOWED_TYPES:
            errors.append(
                f"input_schema.inputs[{idx}].field_type must be one of "
                f"{sorted(_INPUT_SCHEMA_ALLOWED_TYPES)}, got {ftype!r}"
            )

        frequired = item.get("field_required")
        if not isinstance(frequired, bool):
            errors.append(
                f"input_schema.inputs[{idx}].field_required must be bool, "
                f"got {type(frequired).__name__}"
            )

        if not item.get("field_description"):
            errors.append(
                f"input_schema.inputs[{idx}] ({name!r}) must have a non-empty field_description"
            )

    return errors


def _extract_raw_input_keys(preprocess_code: str) -> set:
    """Return all field names accessed via `raw_input['...']` in the code."""
    if not preprocess_code:
        return set()
    return set(_RAW_INPUT_KEY_RE.findall(preprocess_code))


def _schema_field_names(schema) -> set:
    """Return the set of declared field_name values from a list-form input_schema."""
    if not isinstance(schema, dict):
        return set()
    inputs = schema.get("inputs")
    if not isinstance(inputs, list):
        return set()
    return {
        item["field_name"] for item in inputs
        if isinstance(item, dict) and isinstance(item.get("field_name"), str)
    }


def _schema_required_field_names(schema) -> set:
    """Return the set of field_name values marked required=True."""
    if not isinstance(schema, dict):
        return set()
    inputs = schema.get("inputs")
    if not isinstance(inputs, list):
        return set()
    return {
        item["field_name"] for item in inputs
        if isinstance(item, dict)
        and isinstance(item.get("field_name"), str)
        and item.get("field_required") is True
    }


def _check_preprocess_schema_consistency(preprocess_code: str, schema: dict):
    """Check that preprocess code and schema agree on the set of input fields.

    Returns (code_keys, error_list). code_keys is the set extracted from
    `raw_input[...]` accesses. error_list is empty when consistent - same
    rules used later by fix_runtime_code for schema drift recovery:
      * code_keys - schema_keys : fatal (code reads undeclared fields)
      * schema_required_keys - code_keys : fatal (declared required but unused)
      * schema_optional_keys - code_keys : OK (optional field not read)
    """
    code_keys = _extract_raw_input_keys(preprocess_code)
    schema_keys = _schema_field_names(schema)
    schema_required_keys = _schema_required_field_names(schema)

    errors = []
    unknown = code_keys - schema_keys
    if unknown:
        errors.append(
            f"preprocess reads raw_input[...] keys not declared in schema: "
            f"{sorted(unknown)}"
        )

    unused_required = schema_required_keys - code_keys
    if unused_required:
        errors.append(
            f"schema declares required fields that preprocess never reads: "
            f"{sorted(unused_required)}"
        )

    return code_keys, errors


class APIAdaptation(BasePhase):
    name: str = "APIAdaptation"
    description: str = "Generate inference pipeline code (load_model, preprocess, inference, postprocess) that wraps the repository's model entry point into a single-sample FastAPI service. Code generation is required to match the model's input/output contract."
    goal: str = "produce a working service.py with generated pipeline code that correctly matches the model's interface for single-sample inference. REQUIRED EXECUTION ORDER: (1) find_entry -> (2) find_run_command -> (3) establish_fastapi_app -> (4) install_fastapi_package -> (5) infer_io_contract -> (6) adapt_config_file -> (7) adapt_code. Steps 1-4 are COMPULSORY prerequisites that MUST be completed before code generation."
    tools_schemas: List[Dict[str, any]] = [
        {"name": "find_entry", "description": "Find the most likely inference entry file from the repository, README, and deployment task. Use this before trying to infer the run command or related inference files.", "args": {"repo_root": Path, "cleaned_readmes_path": Path, "task": str}},
        {"name": "find_run_command", "description": "Infer the most likely inference-time run command from the entry file, deployment task, and README. Use this after the entry file has been identified.", "args": {"entry_point_path": Path, "task": str, "cleaned_readmes_path": Path}},
        {"name": "check_container_status", "description": "Verify that the Docker container produced by DockerSetUp is still running. Call this before any tool that touches the container (install_fastapi_package and any other docker exec call). requires: container_id. produces: container_status (bool).", "args": {"container_id": str}},
        {"name": "establish_fastapi_app", "description": "(COMPULSORY) Copy the FastAPI app template into the repository and wire the import so app.py can call the service pipeline. MUST be called before install_fastapi_package and before adapt_code. Produces the permanent variable fastapiapp_dir, which is required later by generate_api_documentation.", "args": {"repo_root": Path, "entry_point_dir": Path}},
        {"name": "install_fastapi_package", "description": "(COMPULSORY) Install FastAPI, uvicorn, and python-multipart in the container. MUST be called after establish_fastapi_app. requires: container_id and a running container - call check_container_status first; if it reports false, ask DockerSetUp to recreate the container before retrying.", "args": {"container_id": str}},
        {"name": "get_file_tree", "description": "Return the file tree for a directory or file path so later steps can reason about available code and config candidates.", "args": {"path": Path}},
        # {"name": "read_file", "description": "Carefully use this tool. You can use it when you HAVE TO. Read a file when you need its exact source content for analysis, path validation, or code adaptation.", "args": {"file_path": Path}},
        # {"name": "analyze_run_command", "description": "Analyze the inferred run command and entry file to identify high-confidence candidate files for preprocess, load_model, inference, postprocess, and config.", "args": {"repo_root": Path, "task": str, "run_command": str, "entry_point_path": Path}},
        {"name": "infer_io_contract", "description": "Infer a conservative single-sample inference input and output contract from the task, README, entry point path, and entry point code. Use this before adapt_code as an internal contract hypothesis.", "args": {"task": str, "cleaned_readmes_path": Path, "entry_point_path": Path, "run_command": str}},
        {"name": "adapt_config_file", "description": "Generate a deployment config for single-sample inference using the original repo config files, the weights path, the directory containing downloaded weight-related files (weights, tokenizer, config, vocab, etc.), and the entry file behavior. MUST be called before adapt_code — it writes adapted_config.json and stores adapted_config_path as a permanent variable that adapt_code requires.", "args": {"repo_root": Path, "weights_path": Path, "weights_dir": Path, "entry_point_path": Path, "config_files": List[str]}},
        {"name": "adapt_code", "description": "(COMPULSORY) Generate a `service.py` inference pipeline for single-sample inference, using the inferred I/O contract to guide the four pipeline stages: load_model, preprocess, inference, and postprocess, then verify and save the result. MUST be called AFTER adapt_config_file — pass the permanent variable adapted_config_path produced by adapt_config_file as the adapted_config_path argument. Calling this tool without first calling adapt_config_file will fail. Related source files are resolved automatically from entry_point imports — no need to specify them manually.", "args": {"repo_root": Path, "task": str, "weights_path": Path, "entry_point_path": Path, "run_command": str, "adapted_config_path": Path, "io_contract": Dict[str, Any], "container_id": str}},
    ]
    allowed_parallel_phases: List[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.backend = "gr"
        self.code_backend = "gr"
        self.root_folder = Path(__file__).resolve().parent.parent
        self.tools = {
            "find_entry": self.find_entry,
            "find_run_command": self.find_run_command,
            "check_container_status": self.check_container_status,
            "establish_fastapi_app": self.establish_fastapi_app,
            "install_fastapi_package": self.install_fastapi_package,
            "get_file_tree": self.get_file_tree,
            # "read_file": self.read_file,
            # "analyze_run_command": self.analyze_run_command,
            "infer_io_contract": self.infer_io_contract,
            "adapt_config_file": self.adapt_config_file,
            "adapt_code": self.adapt_code,
        }

        prompt_path = os.path.join(project_path, "prompts")

        run_command_prompt_filepath = os.path.join(prompt_path, "run_command_prompt.json")
        with open(run_command_prompt_filepath, "r", encoding="utf-8") as f:
            self.run_command_prompt = json.load(f)

        run_command_analyze_prompt_filepath = os.path.join(prompt_path, "run_command_analyze_prompt.json")
        with open(run_command_analyze_prompt_filepath, "r", encoding="utf-8") as f:
            self.run_command_analyze_prompt = json.load(f)

        path_checker_prompt_filepath = os.path.join(prompt_path, "path_checker_prompt.json")
        with open(path_checker_prompt_filepath, "r", encoding="utf-8") as f:
            self.path_checker_prompt = json.load(f)

        adapt_config_prompt_filepath = os.path.join(prompt_path, "adapt_config_prompt.json")
        with open(adapt_config_prompt_filepath, "r", encoding="utf-8") as f:
            self.adapt_config_prompt = json.load(f)

        adapt_code_prompt_filepath = os.path.join(prompt_path, "adapt_code_prompt.json")
        with open(adapt_code_prompt_filepath, "r", encoding="utf-8") as f:
            self.adapt_code_prompt = json.load(f)

        verify_service_code_prompt_filepath = os.path.join(prompt_path, "verify_service_code_prompt.json")
        with open(verify_service_code_prompt_filepath, "r", encoding="utf-8") as f:
            self.verify_service_code_prompt = json.load(f)

        infer_io_contract_prompt_filepath = os.path.join(prompt_path, "infer_io_contract_prompt.json")
        with open(infer_io_contract_prompt_filepath, "r", encoding="utf-8") as f:
            self.infer_io_contract_prompt = json.load(f)

    def boundary_tools(self, tool_name: str) -> bool:
        boundary_tools_list = ["adapt_code"]
        if tool_name in boundary_tools_list:
            return True
        else:
            return False

    def tool_arguments(self, tool_name: str) -> Dict[str, any]:
        tool_arguments_dict = {
            "find_entry": {"repo_root": Path, "cleaned_readmes_path": Path, "task": str},
            "find_run_command": {"entry_point_path": Path, "task": str, "cleaned_readmes_path": Path},
            "check_container_status": {"container_id": str},
            "establish_fastapi_app": {"repo_root": Path, "entry_point_dir": Path},
            "install_fastapi_package": {"container_id": str},
            "get_file_tree": {"path": Path},
            "read_file": {"file_path": Path},
            # "analyze_run_command": {"repo_root": Path, "task": str, "run_command": str, "entry_point_path": Path},
            "infer_io_contract": {"task": str, "cleaned_readmes_path": Path, "entry_point_path": Path, "run_command": str},
            "adapt_config_file": {"repo_root": Path, "weights_path": Path, "weights_dir": Path, "entry_point_path": Path, "config_files": List[str]},
            "adapt_code": {"repo_root": Path, "task": str, "weights_path": Path, "entry_point_path": Path, "run_command": str, "adapted_config_path": Path, "io_contract": Dict[str, Any], "container_id": str},
        }
        return tool_arguments_dict[tool_name]

    def ipynb2py(self, path: Path):
        try:
            linux_command(f"pip install nbconvert")
            linux_command(f"jupyter nbconvert --to script {path}")
            return True
        except Exception as e:
            return False

    def find_entry(self, repo_root: Path, cleaned_readmes_path: Path, task: str):
        try:
            repo_root = Path(repo_root)
            entry_point = EntryFindTool(repo_root, cleaned_readmes_path, task)
            primary, backups = entry_point.find_entry()

            if not primary:
                return [{
                    "value": "There is no entry point, please create a entry point.",
                    "storage": "error",
                    "variable_name": "no_entry_point"
                }]

            # Try primary then each backup in order. Each path runs through
            # the same ipynb-conversion + locate_path pipeline; the first one
            # that resolves to a unique file wins.
            attempts = []
            for candidate in [primary] + list(backups):
                resolved_candidate = candidate
                if resolved_candidate.endswith(".ipynb"):
                    try:
                        self.ipynb2py(resolved_candidate)
                        resolved_candidate = resolved_candidate.replace(".ipynb", ".py")
                    except Exception as e:
                        attempts.append(f"{candidate}: ipynb2py failed ({e})")
                        continue

                found, located, node_type = locate_path(repo_root, resolved_candidate)
                if found:
                    return [{
                        "value": located,
                        "storage": "permanent",
                        "variable_name": "entry_point_path"
                    },
                    {
                        "value": os.path.dirname(located),
                        "storage": "permanent",
                        "variable_name": "entry_point_dir"
                    }]
                if isinstance(located, list):
                    attempts.append(f"{candidate}: ambiguous, matches={located}")
                else:
                    attempts.append(f"{candidate}: not found")

            return [{
                "value": (
                    f"None of the LLM-suggested entry points resolved uniquely under "
                    f"{repo_root}. Attempts:\n  - " + "\n  - ".join(attempts)
                ),
                "storage": "error",
                "variable_name": "no_entry_point"
            }]
        except Exception as e:
            return [{
                "value": f"Error finding entry point: {str(e)}",
                "storage": "error",
                "variable_name": "find_entry_error"
            }]

    def assemble_run_command_prompt(self, entry_point_path: Path, task: str, cleaned_readmes_path: Path):
        prompt_parts = []
        role_prompt = "\n".join(self.run_command_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        user_task_prompt = "\n".join(self.run_command_prompt['user_task']).format(
            user_task=task
        )
        prompt_parts.append("\n=== USER TASK ===")
        prompt_parts.append(user_task_prompt)
        
        entry_point_path_prompt = "\n".join(self.run_command_prompt['entry_point_path']).format(
            entry_point_path=entry_point_path
        )
        prompt_parts.append("\n=== ENTRY POINT PATH ===")
        prompt_parts.append(entry_point_path_prompt)
        
        with open(cleaned_readmes_path, "r", encoding="utf-8") as f:
            cleaned_readme_content = f.read()
        readme_content_prompt = "\n".join(self.run_command_prompt['readme_content']).format(
            readme_content=cleaned_readme_content
        )
        prompt_parts.append("\n=== README CONTENT ===")
        prompt_parts.append(readme_content_prompt)
        
        rules_prompt = "\n".join(self.run_command_prompt['rules'])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)
        
        output_format_prompt = json.dumps(self.run_command_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)
        
        prompt = "\n".join(prompt_parts)
        return prompt

    def establish_fastapi_app(self, repo_root: Path, entry_point_dir: Path):
        try:
            repo_root = Path(repo_root)
            entry_point_dir = Path(entry_point_dir)
            shutil.copytree(self.root_folder / "prefile/fastapiapp", repo_root / "fastapiapp", dirs_exist_ok=True)

            fastapiapp_dir = repo_root / "fastapiapp"
            import_module = path_to_import(repo_root, entry_point_dir, "service.py")
            app_path = os.path.join(fastapiapp_dir, "app.py")
            target_prefix = "from service import predict, load_model"

            if import_module is not None:
                new_line = f"from {import_module} import predict, load_model"
            else:
                # path contains characters invalid in a Python identifier (e.g. '-'); import dynamically via importlib
                service_file = entry_point_dir / "service.py"
                rel_service = service_file.relative_to(repo_root)
                # inside the container the repo is mounted at /workspace
                new_line = (
                    "import importlib.util as _ilu; "
                    "from pathlib import Path as _P; "
                    f"_spec = _ilu.spec_from_file_location('_service', str(_P(__file__).resolve().parents[1] / '{rel_service}')); "
                    "_mod = _ilu.module_from_spec(_spec); "
                    "_spec.loader.exec_module(_mod); "
                    "predict = _mod.predict; "
                    "load_model = _mod.load_model"
                )

            replace_line(app_path, target_prefix, new_line)

            # Ensure .autodeploy/ exists so subsequent writes
            # (adapt_code -> input_schema.json, fix_runtime_code -> schema
            # updates, FastAPI app.py -> schema re-read at request time) never
            # trip on a missing parent directory. adapt_code already does
            # mkdir(parents=True, exist_ok=True) on its side, but putting the
            # directory creation here as well means the invariant holds even
            # in edge cases where establish_fastapi_app runs before the first
            # adapt_code attempt.
            (Path(repo_root) / ".autodeploy").mkdir(parents=True, exist_ok=True)

            return [{
                "value": str(fastapiapp_dir),
                "storage": "permanent",
                "variable_name": "fastapiapp_dir"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "establish_fastapi_app_error"
            }]

    def check_container_status(self, container_id: str):
        """Liveness check for the container produced by DockerSetUp.

        Shares the inspect_running helper with DockerSetUp.docker_run polling
        and ServiceDelivery.check_container_status, so all three phases agree
        on what "running" means.
        """
        return [{
            "value": inspect_running(container_id),
            "storage": "temporary",
            "variable_name": "container_status"
        }]

    def install_fastapi_package(self, container_id: str):
        try:
            if not inspect_running(container_id):
                return [{
                    "value": (
                        f"Container {container_id} is not running. "
                        "Re-run DockerSetUp.docker_run before retrying install_fastapi_package."
                    ),
                    "storage": "error",
                    "variable_name": "install_fastapi_package_error"
                }]
            install_result = linux_command_in_docker(f"python3 -m pip install fastapi uvicorn python-multipart", container_id)
            if install_result.returncode == 0:
                return [{
                    "value": "Success",
                    "storage": "temporary",
                    "variable_name": "fastapi_package_install_result"
                }]
            return [{
                "value": install_result.stderr,
                "storage": "error",
                "variable_name": "install_fastapi_package_error"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "install_fastapi_package_error"
            }]

    def find_run_command(self, entry_point_path: Path, task: str, cleaned_readmes_path: Path):
        prompt = self.assemble_run_command_prompt(entry_point_path, task, cleaned_readmes_path)
        run_command = json_query(prompt, "run_command", self.backend)
        if isinstance(run_command, str):
            run_command = json.loads(run_command)
        run_command = run_command['run_command']
        return [{
            "value": run_command,
            "storage": "permanent",
            "variable_name": "run_command"
        }]

    def get_file_tree(self, path : Path):
        tree = build_tree(path)
        return [{
            "value": tree,
            "storage": "permanent",
            "variable_name": f"file_tree of {path}"
        }]

    def read_file(self, file_path: Path):
        with open(file_path, "r") as f:
            file_content = f.read()
        return [{
            "value": file_content,
            "storage": "temporary",
            "variable_name": f"content of {file_path}"
        }]

    def assemble_run_command_analyze_prompt(self, task: str, tree: dict, run_command: str, entry_point_path: Path, entry_point_import: List[str]):
        prompt = []
        role_prompt = "\n".join(self.run_command_analyze_prompt['role'])
        prompt.append("\n=== ROLE DEFINITION ===")
        prompt.append(role_prompt)

        system_definition_prompt = "\n".join(self.run_command_analyze_prompt['system_definition'])
        prompt.append("\n=== SYSTEM DEFINITION ===")
        prompt.append(system_definition_prompt)

        instructions_prompt = "\n".join(self.run_command_analyze_prompt['instructions'])
        prompt.append("\n=== INSTRUCTIONS ===")
        prompt.append(instructions_prompt)

        user_task_prompt = self.run_command_analyze_prompt['task'].format(task=task)
        prompt.append("\n=== USER TASK ===")
        prompt.append(user_task_prompt)

        entry_point_path_prompt = self.run_command_analyze_prompt['entry_point_path'].format(entry_point_path=entry_point_path)
        prompt.append("\n=== ENTRY POINT PATH ===")
        prompt.append(entry_point_path_prompt)

        entry_point_import_prompt = self.run_command_analyze_prompt['entry_point_import'].format(entry_point_import=entry_point_import)
        prompt.append("\n=== ENTRY POINT IMPORT ===")
        prompt.append(entry_point_import_prompt)

        run_command_prompt = self.run_command_analyze_prompt['run_command'].format(run_command=run_command)
        prompt.append("\n=== RUN COMMAND ===")
        prompt.append(run_command_prompt)

        file_tree_prompt = self.run_command_analyze_prompt['file_tree'].format(file_tree=tree)
        prompt.append("\n=== FILE TREE ===")
        prompt.append(file_tree_prompt)

        prompt.append("\n=== OUTPUT FORMAT ===")
        prompt.append(json.dumps(self.run_command_analyze_prompt['output_format']))

        prompt = "\n".join(prompt)
        return prompt

    def assemble_path_checker_prompt(self, tree: dict, file_path: str):
        prompt_parts = []

        role_prompt = "\n".join(self.path_checker_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        file_tree_prompt = "\n".join(self.path_checker_prompt['file_tree']).format(file_tree=tree)
        prompt_parts.append("\n=== FILE TREE ===")
        prompt_parts.append(file_tree_prompt)

        target_path_prompt = "\n".join(self.path_checker_prompt['target_path']).format(target_path=file_path)
        prompt_parts.append("\n=== TARGET PATH ===")
        prompt_parts.append(target_path_prompt)

        output_format_prompt = json.dumps(self.path_checker_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        prompt = "\n".join(prompt_parts)
        return prompt

    def check_analysis_result(self, repo_root: Path, tree: dict, file_list: list[str]):
        repo_root = Path(repo_root)
        checked_file_list = []
        if len(file_list) == 0:
            return []
        for file_path in file_list:
            check_turn = 0
            found, file_path, node_type = locate_path(repo_root, file_path)
            if not found:
                while not found and check_turn < 3:
                    query = self.assemble_path_checker_prompt(tree, file_path)
                    result = json_query(query, "path_checker", self.backend)
                    if isinstance(result, str):
                        result = json.loads(result)
                    file_path = result['file_path']
                    found, file_path, node_type = locate_path(repo_root, file_path)
                    check_turn += 1
            if found:
                checked_file_list.append(file_path)
        return checked_file_list
        
    def analyze_run_command(self, repo_root: Path, task: str, run_command: str, entry_point_path: Path):
        try:
            repo_root = Path(repo_root)
            found, entry_point_path, node_type = locate_path(repo_root, entry_point_path)
            entry_point_import = get_local_import_lines(entry_point_path, Path(entry_point_path).parent)
            tree = build_tree(Path(entry_point_path).parent)
            prompt = self.assemble_run_command_analyze_prompt(task, tree, run_command, entry_point_path, entry_point_import)
            result = json_query(prompt, "run_command_analyze", self.backend)
            if isinstance(result, str):
                result = json.loads(result)
            preprocess_files = self.check_analysis_result(repo_root, tree, result['preprocess_files'])
            load_model_files = self.check_analysis_result(repo_root, tree, result['load_model_files'])
            inference_files = self.check_analysis_result(repo_root, tree, result['inference_files'])
            postprocess_files = self.check_analysis_result(repo_root, tree, result['postprocess_files'])
            config_files = self.check_analysis_result(repo_root, tree, result['config_files'])
            return [{
                "value": preprocess_files,
                "storage": "temporary",
                "variable_name": "preprocess_files"
            },
            {
                "value": load_model_files,
                "storage": "temporary",
                "variable_name": "load_model_files"
            },
            {
                "value": inference_files,
                "storage": "temporary",
                "variable_name": "inference_files"
            },
            {
                "value": postprocess_files,
                "storage": "temporary",
                "variable_name": "postprocess_files"
            },
            {
                "value": config_files,
                "storage": "temporary",
                "variable_name": "config_files"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "analyze_run_command_error"
            }]

    def assemble_infer_io_contract_prompt(self, task: str, cleaned_readmes_path: Path, entry_point_path: Path, run_command: str) -> str:
        prompt_parts = []

        role_prompt = "\n".join(self.infer_io_contract_prompt["role"])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        instructions_prompt = "\n".join(self.infer_io_contract_prompt["instructions"])
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append(instructions_prompt)

        rules_prompt = "\n".join(self.infer_io_contract_prompt["rules"])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)

        task_prompt = "\n".join(self.infer_io_contract_prompt["task"]).format(task=task)
        prompt_parts.append("\n=== TASK ===")
        prompt_parts.append(task_prompt)

        with open(cleaned_readmes_path, "r", encoding="utf-8", errors="ignore") as f:
            readme_content = f.read()
        readme_prompt = "\n".join(self.infer_io_contract_prompt["readme_content"]).format(
            readme_content=readme_content
        )
        prompt_parts.append("\n=== README CONTENT ===")
        prompt_parts.append(readme_prompt)

        with open(entry_point_path, "r", encoding="utf-8", errors="ignore") as f:
            entry_point_content = f.read()
        entry_point_prompt = "\n".join(self.infer_io_contract_prompt["entry_point"]).format(
            entry_point_path=entry_point_path,
            entry_point_content=entry_point_content,
        )
        prompt_parts.append("\n=== ENTRY POINT ===")
        prompt_parts.append(entry_point_prompt)

        run_command_prompt = "\n".join(self.infer_io_contract_prompt["run_command"]).format(
            run_command=run_command
        )
        prompt_parts.append("\n=== RUN COMMAND ===")
        prompt_parts.append(run_command_prompt)

        output_format_prompt = json.dumps(self.infer_io_contract_prompt["output_format"])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        return "\n".join(prompt_parts)

    def infer_io_contract(self, task: str, cleaned_readmes_path: Path, entry_point_path: Path, run_command: str):
        try:
            prompt = self.assemble_infer_io_contract_prompt(
                task=task,
                cleaned_readmes_path=cleaned_readmes_path,
                entry_point_path=entry_point_path,
                run_command=run_command,
            )
            result = json_query(prompt, "infer_io_contract", self.backend)
            if isinstance(result, str):
                result = json.loads(result)
            return [{
                "value": result,
                "storage": "permanent",
                "variable_name": "io_contract"
            },
            {
                "value": result.get("input_type", "unknown"),
                "storage": "temporary",
                "variable_name": "input_type"
            },
            {
                "value": result.get("output_type", "unknown"),
                "storage": "temporary",
                "variable_name": "output_type"
            },
            {
                "value": result.get("confidence", "low"),
                "storage": "temporary",
                "variable_name": "io_contract_confidence"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "infer_io_contract_error"
            }]

    def _normalize_config_files(self, config_files) -> List[str]:
        """Accept the many shapes LLM/upstream may hand us and return a clean
        list of existing config file paths.

        Handles: None, [], real list, JSON/repr-string of a list (e.g. "[]",
        "['a.yaml']"), single string path. Silently drops entries that don't
        resolve to a readable file - the caller's job is to generate a config
        even when no inputs are available, so a bad entry shouldn't abort it.
        """
        if not config_files:
            return []
        if isinstance(config_files, str):
            s = config_files.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    parsed = [s]
            config_files = parsed
        if not isinstance(config_files, (list, tuple)):
            config_files = [config_files]
        return [str(p) for p in config_files if p and os.path.isfile(str(p))]

    def assemble_adapt_config_file_prompt(self, repo_root: Path, weights_path: Path, weights_dir: Path, entry_point_path: Path, config_files: List[str]):
        prompt_parts = []

        role_prompt = "\n".join(self.adapt_config_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        instructions_prompt = "\n".join(self.adapt_config_prompt['instructions'])
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append(instructions_prompt)

        field_definition_prompt = "\n".join(self.adapt_config_prompt['field_definition'])
        prompt_parts.append("\n=== FIELD DEFINITION ===")
        prompt_parts.append(field_definition_prompt)

        with open(entry_point_path, "r", encoding="utf-8") as f:
            entry_point_content = f.read()
        entry_point_prompt = "\n".join(self.adapt_config_prompt['entry_point']).format(entry_point_path=entry_point_path, entry_point_content=entry_point_content)
        prompt_parts.append("\n=== ENTRY POINT ===")
        prompt_parts.append(entry_point_prompt)

        weights_path_prompt = "\n".join(self.adapt_config_prompt['weights_path']).format(weights_path=weights_path)
        prompt_parts.append("\n=== WEIGHTS PATH ===")
        prompt_parts.append(weights_path_prompt)

        weights_dir_path = Path(weights_dir) if weights_dir else None
        if weights_dir_path and weights_dir_path.is_dir():
            entries = sorted(
                str(p.relative_to(weights_dir_path))
                for p in weights_dir_path.rglob("*")
                if p.is_file()
            )
            weights_dir_contents = "\n".join(f"- {e}" for e in entries) if entries else "(directory is empty)"
        else:
            weights_dir_contents = "(directory does not exist or was not provided)"
        weights_dir_prompt = "\n".join(self.adapt_config_prompt['weights_dir']).format(
            weights_dir=weights_dir,
            weights_dir_contents=weights_dir_contents,
        )
        prompt_parts.append("\n=== WEIGHTS DIR ===")
        prompt_parts.append(weights_dir_prompt)

        file_tree = build_tree(repo_root)
        file_tree_prompt = "\n".join(self.adapt_config_prompt['file_tree']).format(file_tree=file_tree)
        prompt_parts.append("\n=== FILE TREE ===")
        prompt_parts.append(file_tree_prompt)

        # Filter out binary files that the agent may have mistakenly
        # included as "config files" (weight checkpoints, tokenizer models,
        # etc.). Only text-based config files should be read and fed to the
        # LLM prompt.
        _BINARY_SUFFIXES = {
            ".bin", ".pt", ".pth", ".safetensors", ".ckpt",
            ".model", ".pkl", ".pickle", ".h5", ".hdf5",
            ".onnx", ".tflite", ".pb", ".msgpack",
        }
        normalized_config_files = self._normalize_config_files(config_files)
        normalized_config_files = [
            f for f in normalized_config_files
            if Path(f).suffix.lower() not in _BINARY_SUFFIXES
        ]
        if normalized_config_files:
            config_files_content = []
            for config_file in normalized_config_files:
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config_files_content.append(f"---{config_file}---")
                        config_files_content.append(f.read())
                except UnicodeDecodeError:
                    print(f"[APIAdaptation] skipping binary file: {config_file}")
                    continue
            config_files_content_prompt = "\n".join(self.adapt_config_prompt['config_files_content']).format(config_file_content="\n".join(config_files_content))
        else:
            # No config files in the repo (or none could be located). Tell the
            # LLM explicitly instead of formatting an empty list into the
            # template - an empty "[]" confuses the model and a missing
            # section trips the template's format() call.
            config_files_content_prompt = "No config files were provided. Generate an adapted deployment config from the entry point, weights path, and file tree alone."
        prompt_parts.append("\n=== CONFIG FILES CONTENT ===")
        prompt_parts.append(config_files_content_prompt)

        output_format_prompt = json.dumps(self.adapt_config_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        prompt = "\n".join(prompt_parts)
        return prompt

    def parse_maybe_dict(self, x):
        if isinstance(x, dict):
            return x
        if not isinstance(x, str):
            return {}

        x = x.strip()
        if not x:
            return {}

        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(x)
            except Exception:
                return {}

    @staticmethod
    def _remap_paths_to_container(obj, repo_root_str: str, work_dir: str):
        """Recursively replace host repo_root paths with container work_dir paths."""
        if isinstance(obj, str):
            return obj.replace(repo_root_str, work_dir)
        if isinstance(obj, dict):
            return {
                APIAdaptation._remap_paths_to_container(k, repo_root_str, work_dir):
                APIAdaptation._remap_paths_to_container(v, repo_root_str, work_dir)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [APIAdaptation._remap_paths_to_container(item, repo_root_str, work_dir) for item in obj]
        return obj

    def adapt_config_file(self, repo_root: Path, weights_path: Path, weights_dir: Path, entry_point_path: Path, config_files: List[str]):
        try:
            prompt = self.assemble_adapt_config_file_prompt(repo_root, weights_path, weights_dir, entry_point_path, config_files)
            adapted_config = code_query(prompt, "adapt_config_file", self.code_backend)
            config = yaml.safe_load(adapted_config["config"]) if isinstance(adapted_config["config"], str) else adapted_config["config"]
            path_mapping = self.parse_maybe_dict(adapted_config["path_mapping"])
            key_mapping = self.parse_maybe_dict(adapted_config["key_mapping"])

            # Remap host repo_root paths to container work_dir so that
            # service.py (which runs inside the container) gets correct paths
            # from the start, avoiding a MODEL_LOAD_ERROR -> fix_runtime_code
            # round-trip.
            docker_setting = global_config.get("docker_setting") or {}
            work_dir = str(docker_setting.get("work_dir") or "/workspace")
            repo_root_str = str(Path(repo_root).resolve())
            config = self._remap_paths_to_container(config, repo_root_str, work_dir)
            path_mapping = self._remap_paths_to_container(path_mapping, repo_root_str, work_dir)
            print(
                f"[APIAdaptation] adapt_config_file: remapped host paths "
                f"'{repo_root_str}' -> '{work_dir}' in config and path_mapping"
            )

            clean_data = {
                "config": config,
                "path_mapping": path_mapping,
                "key_mapping": key_mapping
            }
            adapted_config_path = os.path.join(Path(repo_root), ".autodeploy", "adapted_config.json")
            with open(adapted_config_path, "w", encoding="utf-8") as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)
            return [{
                "value": adapted_config_path,
                "storage": "permanent",
                "variable_name": "adapted_config_path"
            },
            {
                "value": "The config file is successfully adapted.",
                "storage": "temporary",
                "variable_name": "adapt_config_result"
            }]
        except Exception as e:
            return [{
                "value": str(e),
                "storage": "error",
                "variable_name": "adapt_config_file_error"
            }]

    def assemble_adapt_code_prompt(self, func_name: str, task: str, weights_path: Path, entry_point_path: Path, run_command: str, adapted_config_path: Path, io_contract: Dict[str, Any], related_code_files: List[str], installed_packages: str = "", weights_dir_listing: str = "", repo_file_tree: str = ""):
        prompt_parts = []

        func_prompt = json.dumps(self.adapt_code_prompt[func_name])
        func_prompt = json.loads(func_prompt)
        role_prompt = "\n".join(func_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)
        instructions_prompt = "\n".join(func_prompt['instructions'])
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append(instructions_prompt)
        fuction_return_format_prompt = "\n".join(func_prompt['fuction_return_format'])
        prompt_parts.append("\n=== FUNCTION RETURN FORMAT ===")
        prompt_parts.append(fuction_return_format_prompt)

        pipeline_contract_prompt = "\n".join(self.adapt_code_prompt['pipeline_contract'])
        prompt_parts.append("\n=== PIPELINE CONTRACT ===")
        prompt_parts.append(pipeline_contract_prompt)


        with open(adapted_config_path, "r", encoding="utf-8") as f:
            adapted_config = json.load(f)
        config_file_content_prompt = "\n".join(self.adapt_code_prompt['config_file']).format(config_file_path=adapted_config_path, config_file_content=adapted_config["config"])
        prompt_parts.append("\n=== CONFIG FILE CONTENT ===")
        prompt_parts.append(config_file_content_prompt)

        with open(entry_point_path, "r", encoding="utf-8") as f:
            entry_point_content = f.read()
        entry_point_path_prompt = "\n".join(self.adapt_code_prompt['entry_point']).format(entry_point_path=entry_point_path, entry_point_content=entry_point_content)
        prompt_parts.append("\n=== ENTRY POINT===")
        prompt_parts.append(entry_point_path_prompt)

        run_command_prompt = "\n".join(self.adapt_code_prompt['run_command']).format(run_command=run_command)
        prompt_parts.append("\n=== RUN COMMAND ===")
        prompt_parts.append(run_command_prompt)

        weights_path_prompt = "\n".join(self.adapt_code_prompt['weights_path']).format(weights_path=weights_path)
        prompt_parts.append("\n=== WEIGHTS PATH ===")
        prompt_parts.append(weights_path_prompt)

        user_task_prompt = "\n".join(self.adapt_code_prompt['user_task']).format(task=task)
        prompt_parts.append("\n=== USER TASK ===")
        prompt_parts.append(user_task_prompt)

        io_contract_prompt = "\n".join(self.adapt_code_prompt['io_contract']).format(
            io_contract=json.dumps(io_contract, ensure_ascii=False)
        )
        prompt_parts.append("\n=== INFERRED IO CONTRACT ===")
        prompt_parts.append(io_contract_prompt)

        
        related_code_files = [Path(related_code_file) for related_code_file in related_code_files if related_code_file != str(entry_point_path)]
        if related_code_files != []:
            _MAX_RELATED_FILES = 10
            _MAX_FILE_CHARS = 5000
            related_code_content = []
            for related_code_file in related_code_files[:_MAX_RELATED_FILES]:
                try:
                    with open(related_code_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    related_code_content.append(f"\n---{related_code_file}---\n")
                    if len(content) > _MAX_FILE_CHARS:
                        related_code_content.append(content[:_MAX_FILE_CHARS])
                        related_code_content.append(f"\n... (truncated, {len(content)} chars total)")
                    else:
                        related_code_content.append(content)
                except Exception:
                    continue
            if len(related_code_files) > _MAX_RELATED_FILES:
                related_code_content.append(f"\n... ({len(related_code_files) - _MAX_RELATED_FILES} more files omitted)")
            related_code_content_prompt = "\n".join(self.adapt_code_prompt['related_code']).format(related_code_content="\n".join(related_code_content))
            prompt_parts.append("\n=== RELATED CODE CONTENT ===")
            prompt_parts.append(related_code_content_prompt)

        # Installed packages inside the container — lets the LLM pick API
        # calls and import paths compatible with the actual runtime versions.
        if installed_packages:
            prompt_parts.append("\n=== INSTALLED PACKAGES (container) ===")
            prompt_parts.append(
                "The following Python packages (with exact versions) are installed "
                "in the deployment container. Your generated code MUST be compatible "
                "with these versions. Do NOT use APIs, arguments, or behaviors that "
                "only exist in newer or older versions than what is listed here."
            )
            prompt_parts.append(installed_packages)

        if weights_dir_listing:
            prompt_parts.append("\n=== WEIGHTS DIRECTORY LISTING ===")
            prompt_parts.append(
                "Files actually present in the container's weights directory. "
                "Use this to determine the correct weight format (.safetensors, .bin, .pt, etc.) "
                "and to find companion files (config.json, tokenizer.json, vocab.txt, etc.)."
            )
            prompt_parts.append(weights_dir_listing)

        if repo_file_tree:
            prompt_parts.append("\n=== REPOSITORY FILE TREE ===")
            prompt_parts.append(
                "Module structure of the repository. Use this to derive correct "
                "Python import paths (e.g. if the tree shows barcodebert/io.py, "
                "the import is `from barcodebert.io import ...`). "
                "Do NOT guess import paths — look them up here."
            )
            prompt_parts.append(repo_file_tree)

        # Stage-local output_format overrides the shared one. Preprocess defines
        # its own with an `input_schema` field; load_model / inference /
        # postprocess fall back to the shared {generated_code, reason} envelope.
        if func_name == "preprocess":
            stage_output_format = func_prompt.get('output_format')
        else:
            stage_output_format = self.adapt_code_prompt.get('output_format')
        if stage_output_format is not None:
            output_format_prompt = json.dumps(stage_output_format, ensure_ascii=False, indent=2)
            prompt_parts.append("\n=== OUTPUT FORMAT ===")
            prompt_parts.append(output_format_prompt)

        prompt = "\n".join(prompt_parts)
        return prompt

    @staticmethod
    def _strip_server_startup(code: str) -> str:
        """Remove any server startup blocks that the LLM may have appended.

        Targets patterns like:
            if __name__ == "__main__":
                uvicorn.run(...)
        These cause PORT_ERROR when service.py is imported by app.py while
        a separate uvicorn process is already binding the same port.
        """
        import re
        # Remove if __name__ == "__main__": blocks at the end of the code
        code = re.sub(
            r'\n*if\s+__name__\s*==\s*["\']__main__["\']\s*:.*',
            '', code, flags=re.DOTALL,
        )
        return code

    def save_inference_pipeline(self, save_path: str, load_model_code: str, preprocess_code: str, inference_code: str, postprocess_code: str):
        """
        Save the code generated by the four stages as one complete Python file.
        """
        # Defensive cleanup: strip any server startup code the LLM may have added
        load_model_code = self._strip_server_startup(load_model_code)
        preprocess_code = self._strip_server_startup(preprocess_code)
        inference_code = self._strip_server_startup(inference_code)
        postprocess_code = self._strip_server_startup(postprocess_code)

        file_text = "\n".join([
            "from typing import Any, Dict, Optional",
            "",
            "_MODEL_BUNDLE: Optional[Dict[str, Any]] = None",
            "",
            load_model_code.strip(),
            "",
            preprocess_code.strip(),
            "",
            inference_code.strip(),
            "",
            postprocess_code.strip(),
            "",
            "def init_model() -> Dict[str, Any]:",
            "    global _MODEL_BUNDLE",
            "    if _MODEL_BUNDLE is None:",
            "        _MODEL_BUNDLE = load_model()",
            "    return _MODEL_BUNDLE",
            "",
            "def predict(raw_input: Any, model_bundle: Optional[Dict[str, Any]] = None) -> Any:",
            "    bundle = model_bundle if model_bundle is not None else init_model()",
            "    model_inputs = preprocess(raw_input, bundle)",
            "    raw_outputs = inference(model_inputs, bundle)",
            "    results = postprocess(raw_outputs, bundle)",
            "    return results",
            "",
        ])

        Path(save_path).write_text(file_text, encoding="utf-8")

    def assemble_verify_service_code_prompt(self, service_code: str):
        prompt_parts = []

        role_prompt = "\n".join(self.verify_service_code_prompt['role'])
        prompt_parts.append("\n=== ROLE DEFINITION ===")
        prompt_parts.append(role_prompt)

        instructions_prompt = "\n".join(self.verify_service_code_prompt['instructions'])
        prompt_parts.append("\n=== INSTRUCTIONS ===")
        prompt_parts.append(instructions_prompt)

        rules_prompt = "\n".join(self.verify_service_code_prompt['rules'])
        prompt_parts.append("\n=== RULES ===")
        prompt_parts.append(rules_prompt)

        pipeline_constraints_prompt = "\n".join(self.verify_service_code_prompt['pipeline_constraints'])
        prompt_parts.append("\n=== PIPELINE CONSTRAINTS ===")
        prompt_parts.append(pipeline_constraints_prompt)

        input_prompt = "\n".join(self.verify_service_code_prompt['input']).format(service_code=service_code)
        prompt_parts.append("\n=== INPUT ===")
        prompt_parts.append(input_prompt)

        output_format_prompt = json.dumps(self.verify_service_code_prompt['output_format'])
        prompt_parts.append("\n=== OUTPUT FORMAT ===")
        prompt_parts.append(output_format_prompt)

        prompt = "\n".join(prompt_parts)
        return prompt

    def adapt_code(self, repo_root: Path, task: str, weights_path: Path, entry_point_path: Path, run_command: str, adapted_config_path: Path, io_contract: Dict[str, Any], container_id: str):
        try:
            # Guard: adapted_config_path must exist and be non-empty.
            # adapt_config_file MUST be called first to produce this file.
            # Catching this early gives a clear, actionable error instead of
            # an opaque JSONDecodeError or FileNotFoundError buried in prompt assembly.
            _config_p = Path(adapted_config_path)
            if not _config_p.exists():
                return [{
                    "value": (
                        f"adapted_config_path '{adapted_config_path}' does not exist. "
                        "adapt_config_file MUST be called before adapt_code to produce this file. "
                        "Call adapt_config_file first, then retry adapt_code with the returned adapted_config_path."
                    ),
                    "storage": "error",
                    "variable_name": "adapt_code_error",
                }]
            if _config_p.stat().st_size == 0:
                return [{
                    "value": (
                        f"adapted_config_path '{adapted_config_path}' exists but is empty. "
                        "adapt_config_file likely failed on the previous call. "
                        "Re-run adapt_config_file to regenerate the config, then retry adapt_code."
                    ),
                    "storage": "error",
                    "variable_name": "adapt_code_error",
                }]

            io_contract = self.parse_maybe_dict(io_contract)

            # Resolve the full import closure starting from entry_point_path.
            # This gives every stage the complete set of locally-imported
            # source files so the LLM can inline logic instead of guessing
            # import paths or calling repo utility functions.
            repo_root = Path(repo_root)
            related_files = resolve_import_closure(
                [str(entry_point_path)], repo_root
            )
            print(
                f"[APIAdaptation] adapt_code: resolved import closure "
                f"({len(related_files)} files) from {entry_point_path}"
            )

            # Remap weights_path to container path so the LLM generates
            # service.py with paths that work inside the Docker container.
            docker_setting = global_config.get("docker_setting") or {}
            container_work_dir = str(docker_setting.get("work_dir") or "/workspace")
            repo_root_str = str(Path(repo_root).resolve())
            container_weights_path = str(weights_path).replace(repo_root_str, container_work_dir)
            print(
                f"[APIAdaptation] adapt_code: remapped weights_path for prompt: "
                f"'{weights_path}' -> '{container_weights_path}'"
            )

            # Query installed package versions from the container so the LLM
            # generates code compatible with the actual runtime environment.
            installed_packages = ""
            try:
                pip_res = linux_command_in_docker(
                    "python3 -m pip list --format=freeze 2>/dev/null", container_id
                )
                if pip_res.returncode == 0 and pip_res.stdout.strip():
                    installed_packages = pip_res.stdout.strip()
                    print(
                        f"[APIAdaptation] adapt_code: captured {len(installed_packages.splitlines())} "
                        f"installed packages from container"
                    )
            except Exception as e:
                print(f"[APIAdaptation] adapt_code: failed to query pip list: {e}")

            # List weight files actually present in the container so the LLM
            # can pick the correct loading method (safetensors vs .bin etc.).
            weights_dir_listing = ""
            try:
                container_weights_dir = str(weights_path).replace(repo_root_str, container_work_dir)
                ls_res = linux_command_in_docker(
                    f"find {container_weights_dir} -type f 2>/dev/null | head -100",
                    container_id,
                )
                if ls_res.returncode == 0 and ls_res.stdout.strip():
                    weights_dir_listing = ls_res.stdout.strip()
                    print(
                        f"[APIAdaptation] adapt_code: weights dir listing "
                        f"({len(weights_dir_listing.splitlines())} files)"
                    )
            except Exception as e:
                print(f"[APIAdaptation] adapt_code: failed to list weights dir: {e}")

            # Build a file tree of the repo so the LLM knows the module
            # structure and can derive correct import paths.
            repo_file_tree = ""
            try:
                tree = build_tree(repo_root, max_depth=3)
                repo_file_tree = json.dumps(tree, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[APIAdaptation] adapt_code: failed to build file tree: {e}")

            load_model_prompt = self.assemble_adapt_code_prompt(func_name="load_model", task=task, weights_path=container_weights_path, entry_point_path=entry_point_path, run_command=run_command, adapted_config_path=adapted_config_path, io_contract=io_contract, related_code_files=related_files, installed_packages=installed_packages, weights_dir_listing=weights_dir_listing, repo_file_tree=repo_file_tree)
            code_res = code_query(load_model_prompt, "adapt_code", self.code_backend)
            if isinstance(code_res, str):
                code_res = json.loads(code_res)
            load_model_code = code_res['generated_code']

            preprocess_prompt = self.assemble_adapt_code_prompt(func_name="preprocess", task=task, weights_path=container_weights_path, entry_point_path=entry_point_path, run_command=run_command, adapted_config_path=adapted_config_path, io_contract=io_contract, related_code_files=related_files, installed_packages=installed_packages, weights_dir_listing=weights_dir_listing, repo_file_tree=repo_file_tree)
            code_res = code_query(preprocess_prompt, "adapt_code_preprocess", self.code_backend)
            if isinstance(code_res, str):
                code_res = json.loads(code_res)
            preprocess_code = code_res['generated_code']

            # The preprocess stage MUST also emit input_schema (see
            # prompts/adapt_code_prompt.json preprocess.output_format). We run
            # two deterministic post-validation passes before accepting it.
            preprocess_input_schema = code_res.get('input_schema')
            if preprocess_input_schema is None:
                return [{
                    "value": (
                        "adapt_code preprocess stage did not emit input_schema. "
                        "The LLM response is missing the 'input_schema' field which "
                        "is mandatory for the preprocess stage - see preprocess.output_format "
                        "in adapt_code_prompt.json."
                    ),
                    "storage": "error",
                    "variable_name": "adapt_code_error",
                }]

            structural_errors = _validate_input_schema_structure(preprocess_input_schema)
            if structural_errors:
                return [{
                    "value": (
                        "adapt_code preprocess input_schema is structurally invalid:\n"
                        + "\n".join(f"  - {e}" for e in structural_errors)
                    ),
                    "storage": "error",
                    "variable_name": "adapt_code_error",
                }]

            _, consistency_errors = _check_preprocess_schema_consistency(
                preprocess_code, preprocess_input_schema
            )
            if consistency_errors:
                code_keys = _extract_raw_input_keys(preprocess_code)
                schema_keys = _schema_field_names(preprocess_input_schema)
                return [{
                    "value": (
                        "adapt_code preprocess code/schema are inconsistent:\n"
                        + "\n".join(f"  - {e}" for e in consistency_errors)
                        + f"\n  preprocess reads raw_input[...] keys: {sorted(code_keys)}"
                        + f"\n  schema declares field_names:          {sorted(schema_keys)}"
                    ),
                    "storage": "error",
                    "variable_name": "adapt_code_error",
                }]

            declared_names = sorted(_schema_field_names(preprocess_input_schema))
            print(
                f"[APIAdaptation] adapt_code: preprocess declared input_schema with "
                f"{len(declared_names)} field(s): {declared_names}"
            )

            inference_prompt = self.assemble_adapt_code_prompt(func_name="inference", task=task, weights_path=container_weights_path, entry_point_path=entry_point_path, run_command=run_command, adapted_config_path=adapted_config_path, io_contract=io_contract, related_code_files=related_files, installed_packages=installed_packages, weights_dir_listing=weights_dir_listing, repo_file_tree=repo_file_tree)
            code_res = code_query(inference_prompt, "adapt_code", self.code_backend)
            if isinstance(code_res, str):
                code_res = json.loads(code_res)
            inference_code = code_res['generated_code']

            postprocess_prompt = self.assemble_adapt_code_prompt(func_name="postprocess", task=task, weights_path=container_weights_path, entry_point_path=entry_point_path, run_command=run_command, adapted_config_path=adapted_config_path, io_contract=io_contract, related_code_files=related_files, installed_packages=installed_packages, weights_dir_listing=weights_dir_listing, repo_file_tree=repo_file_tree)
            code_res = code_query(postprocess_prompt, "adapt_code", self.code_backend)
            if isinstance(code_res, str):
                code_res = json.loads(code_res)
            postprocess_code = code_res['generated_code']


            service_code ="\n".join([
            "from typing import Any, Dict, Optional",
            "",
            "_MODEL_BUNDLE: Optional[Dict[str, Any]] = None",
            "",
            load_model_code.strip(),
            "",
            preprocess_code.strip(),
            "",
            inference_code.strip(),
            "",
            postprocess_code.strip(),
            "",
            "def init_model() -> Dict[str, Any]:",
            "    global _MODEL_BUNDLE",
            "    if _MODEL_BUNDLE is None:",
            "        _MODEL_BUNDLE = load_model()",
            "    return _MODEL_BUNDLE",
            "",
            "def predict(raw_input: Any, model_bundle: Optional[Dict[str, Any]] = None) -> Any:",
            "    bundle = model_bundle if model_bundle is not None else init_model()",
            "    model_inputs = preprocess(raw_input, bundle)",
            "    raw_outputs = inference(model_inputs, bundle)",
            "    results = postprocess(raw_outputs, bundle)",
            "    return results",
            "",
        ])
            verify_service_code_prompt = self.assemble_verify_service_code_prompt(service_code)
            code_res = code_query(verify_service_code_prompt, "verify_service_code", self.code_backend)
            if isinstance(code_res, str):
                code_res = json.loads(code_res)
            verify_service_code_result = code_res
            for _section in ('load_model', 'preprocess', 'inference', 'postprocess'):
                _val = verify_service_code_result[_section]['is_correct']
                if isinstance(_val, str):
                    _val = _val.strip().lower() == "true"
                verify_service_code_result[_section]['is_correct'] = _val
            if not verify_service_code_result['load_model']['is_correct'] and verify_service_code_result['load_model'].get('corrected_code', '').strip():
                load_model_code = verify_service_code_result['load_model']['corrected_code']
            if not verify_service_code_result['preprocess']['is_correct'] and verify_service_code_result['preprocess'].get('corrected_code', '').strip():
                preprocess_code = verify_service_code_result['preprocess']['corrected_code']
            if not verify_service_code_result['inference']['is_correct'] and verify_service_code_result['inference'].get('corrected_code', '').strip():
                inference_code = verify_service_code_result['inference']['corrected_code']
            if not verify_service_code_result['postprocess']['is_correct'] and verify_service_code_result['postprocess'].get('corrected_code', '').strip():
                postprocess_code = verify_service_code_result['postprocess']['corrected_code']

            # Post-process: remap any remaining host paths that the LLM
            # may have picked up from entry_point_path / related_code_files
            # headers in the prompt.  weights_path was already remapped on
            # the input side, but other paths (sys.path.append, open(...))
            # could still reference the host repo root.
            load_model_code = load_model_code.replace(repo_root_str, container_work_dir)
            preprocess_code = preprocess_code.replace(repo_root_str, container_work_dir)
            inference_code = inference_code.replace(repo_root_str, container_work_dir)
            postprocess_code = postprocess_code.replace(repo_root_str, container_work_dir)

            save_path = os.path.join(Path(entry_point_path).parent, "service.py")
            self.save_inference_pipeline(save_path, load_model_code, preprocess_code, inference_code, postprocess_code)

            # Persist input_schema to .autodeploy/input_schema.json via FileTracker
            # so (a) FastAPI can read it at every /predict_text request, and
            # (b) any later fix_runtime_code retry has a versioned history to
            # diff against.
            repo_root = Path(repo_root)
            schema_dir = repo_root / ".autodeploy"
            schema_dir.mkdir(parents=True, exist_ok=True)
            schema_path = schema_dir / "input_schema.json"

            file_tracker = FileTracker(str(repo_root))
            file_tracker.init_workspace()
            schema_version = file_tracker.modify_file(
                str(schema_path),
                json.dumps(preprocess_input_schema, ensure_ascii=False, indent=2),
            )
            print(
                f"[APIAdaptation] adapt_code: wrote {schema_path} (FileTracker v{schema_version})"
            )

            return [{
                "value": str(save_path),
                "storage": "permanent",
                "variable_name": "service_pipeline_path"
            },
            {
                "value": str(schema_path),
                "storage": "permanent",
                "variable_name": "input_schema_path"
            },
            {
                "value": preprocess_input_schema,
                "storage": "permanent",
                "variable_name": "input_schema"
            },
            {
                "value": "The service pipline is successfully saved.",
                "storage": "temporary",
                "variable_name": "service_pipeline_saved"
            }]
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[APIAdaptation] adapt_code failed:\n{tb}")
            return [{
                "value": f"{type(e).__name__}: {e}\n{tb}",
                "storage": "error",
                "variable_name": "adapt_code_error"
            }]
