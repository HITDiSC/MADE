from pydantic import BaseModel
from typing import Any, Dict, Optional, List

class arguments_schema(BaseModel):
    key: str
    value: str

class pm_schema(BaseModel):
    reason: str
    decision: str # ok | blocked | need_retry | fatal
    next_phase: list[str]
    # Phases whose required_outputs are present but proven defective by a
    # downstream failure (e.g. dockersetup's container is broken, revealed only
    # during servicedelivery). Listing a phase here marks it needs_redo in the
    # deployment state so it is re-run instead of trusted. Normally empty.
    invalidate_phases: list[str]

class em_schema(BaseModel):
    reason: str
    tool_name: str
    arguments: list[arguments_schema]

class em_instruction_schema(BaseModel):
    # EM's response to a PM instruction (issued on a need_retry decision).
    # decision: accept -> follow the instruction; negotiate -> push back with a
    # concrete objection/request so PM can revise; reject -> proceed on EM's own
    # plan and report why. `message` carries the objection/justification.
    reason: str
    decision: str  # accept | negotiate | reject
    message: str

class pd_schema(BaseModel):
    reason: str
    phases: list[str]

class pm_gap_schema(BaseModel):
    # PM's mid-phase gap check: should it interrupt EM with a corrective
    # instruction, or let EM keep going? intervene=false when EM is making
    # progress. instruction is the advisory nudge to inject when intervene=true.
    reason: str
    intervene: bool
    instruction: str

class github_link_extract_schema(BaseModel):
    github_link: str

class RangeItem(BaseModel):
    start_line_number: int
    end_line_number: int

class clean_readme_schema(BaseModel):
    keep_ranges: list[RangeItem]

class TaskItem(BaseModel):
    task_id: str
    task_name: str
    task_description: str

class task_selection_schema(BaseModel):
    tasks: list[TaskItem]

class local_dockerfile_find_schema(BaseModel):
    reason: str
    find_dockerfile: bool
    dockerfile_path: str

class dockerfile_generation_schema(BaseModel):
    reason: str
    dockerfile_content: str

class entry_find_schema(BaseModel):
    reason: str
    entry_point: str
    backup_path: List[str]

class local_weights_find_schema(BaseModel):
    reason: str
    find_local_weights: bool
    local_weights_path: str

class run_command_schema(BaseModel):
    reason: str
    run_command: str

class run_command_analyze_schema(BaseModel):
    reason: str
    preprocess_files: list[str]
    load_model_files: list[str]
    inference_files: list[str]
    postprocess_files: list[str]
    config_files: list[str]

class path_checker_schema(BaseModel):
    reason: str
    file_path: str

class adapt_config_file_schema(BaseModel):
    config: str
    path_mapping: str
    key_mapping: str

class adapt_code_schema(BaseModel):
    reason: str
    generated_code: str

class input_schema_item_schema(BaseModel):
    # Each schema field is serialized as a flat object so OpenAI structured
    # output can close over it. `field_name` acts as the key (we later
    # convert the list to a dict on read when a lookup is cheaper).
    field_name: str
    field_type: str            # "string" | "number" | "integer" | "boolean"
    field_required: bool
    field_description: str

class input_schema_schema(BaseModel):
    # NOTE: `params` was intentionally removed (user decision). Runtime
    # parameters do not belong in the input schema; if a model needs them
    # they can be baked into service.py as constants, or added back later
    # as a separate params_schema.
    reason: str
    version: str
    task_type: str
    inputs: list[input_schema_item_schema]

class adapt_code_preprocess_schema(BaseModel):
    reason: str
    generated_code: str
    input_schema: input_schema_schema

class fix_input_schema_schema(BaseModel):
    reason: str
    input_schema: input_schema_schema

class compose_request_body_item_schema(BaseModel):
    """One (schema field -> test case filename) pair returned by the matcher."""
    field_name: str
    filename: str

class compose_request_body_schema(BaseModel):
    """Output envelope for ServiceDelivery._compose_request_body.

    The matcher LLM sees the input_schema fields plus the listing of files
    under case_dir/input/ and decides which file's content should fill each
    schema field. `mapping` is a list of {field_name, filename} pairs. If
    the LLM cannot confidently match every required field, it returns an
    empty list and explains in `reason` - the caller then raises a clear
    CONTRACT_ERROR instead of sending a broken request to FastAPI.
    """
    reason: str
    mapping: list[compose_request_body_item_schema]


class resolve_package_version_schema(BaseModel):
    reason: str
    package_spec: str
    install_flags: str

class analyze_error_schema(BaseModel):
    reason: str
    error_classification: str
    error_suggestion: str

class verify_code(BaseModel):
    is_correct: bool
    corrected_code: str

class verify_service_code_schema(BaseModel):
    preprocess: verify_code
    load_model: verify_code
    inference: verify_code
    postprocess: verify_code

class output_format_schema(BaseModel):
    field_name: str
    field_value: str

class field_judge_schema(BaseModel):
    field_name: str
    field_judge_type: str

class infer_io_contract_schema(BaseModel):
    input_type: str
    input_format: str
    output_type: str
    output_format: str
    request_schema_hint: str
    response_schema_hint: str
    confidence: str
    evidence: list[str]
    notes: str

class cross_function_alignment_schema(BaseModel):
    reason: str
    needs_full_rewrite: bool
    issues: list[str]
    summary: str

class llm_normalize_schema(BaseModel):
    reason: str
    normalized_expected: list[output_format_schema]
    normalized_actual: list[output_format_schema]
    field_judges: list[field_judge_schema]

def output_schema(role):
    schema_map = {
        "pm": pm_schema,
        "em": em_schema,
        "em_instruction": em_instruction_schema,
        "pm_gap": pm_gap_schema,
        "pd": pd_schema,
        "github_link_extract": github_link_extract_schema,
        "clean_readme": clean_readme_schema,
        "task_selection": task_selection_schema,
        "local_dockerfile_find": local_dockerfile_find_schema,
        "dockerfile_generation": dockerfile_generation_schema,
        "entry_find": entry_find_schema,
        "local_weights_find": local_weights_find_schema,
        "run_command": run_command_schema,
        "run_command_analyze": run_command_analyze_schema,
        "path_checker": path_checker_schema,
        "adapt_config_file": adapt_config_file_schema,
        "infer_io_contract": infer_io_contract_schema,
        "adapt_code": adapt_code_schema,
        "adapt_code_preprocess": adapt_code_preprocess_schema,
        "resolve_package_version": resolve_package_version_schema,
        "analyze_error": analyze_error_schema,
        "verify_service_code": verify_service_code_schema,
        "llm_normalize": llm_normalize_schema,
        "compose_request_body": compose_request_body_schema,
        "fix_input_schema": fix_input_schema_schema,
        "cross_function_alignment": cross_function_alignment_schema,
    }
    return schema_map[role]

def role_description_text(role):
    description_map = {
        "pm": "You are the Phase Manager (PM) Agent responsible for deciding the next phase to execute.",
        "em": "You are the Execution Master (EM) Agent responsible for executing the deployment-related actions.",
        "em_instruction": "You are the Execution Master (EM) Agent. The Phase Manager has reviewed your work and issued an instruction. Using your own local observations from the execution history, decide whether to ACCEPT, NEGOTIATE, or REJECT it.",
        "pm_gap": "You are the Phase Manager (PM) Agent monitoring the Execution Master MID-phase. Decide whether EM is stuck and needs a corrective instruction, or is making progress and should be left to continue.",
        "pd": "You are the Parallel Discriminator (PD) Agent responsible for determining whether next phases can be run in parallel.",
        "github_link_extract": "Your task is to extract the Github Link from the PDF file.",
        "clean_readme": "Your task is to clean the readme file and return the keep ranges.",
        "task_selection": "Your task is to select the task for inference in the readme file.",
        "local_dockerfile_find": "Your task is to find the most likely dockerfile path in a repository.",
        "entry_find": "Your task is to find the most likely inference entry-point file in a repository.",
        "dockerfile_generation": "Your task is to generate a Dockerfile for a repository.",
        "local_weights_find": "Your task is to find the most likely local weights path in a repository.",
        "run_command": "Your task is to find the run command to complete the deployment of the model.",
        "run_command_analyze": "Your task is to analyze a repository and select the minimal set of files required to construct a single-sample inference pipeline.",
        "path_checker": "Your task is to find the most likely real path in a file tree for a given target path.",
        "adapt_config_file": "You are an expert in machine learning repository understanding and deployment adaptation.",
        "adapt_code": "Your task is to adapt the code to the task.",
        "infer_io_contract": "Your task is to infer the input and output contract for the deployed inference API.",
        "adapt_code_preprocess": "Your task is to adapt the code to the task.",
        "resolve_package_version": "Your task is to determine the best compatible version of a Python package based on the Dockerfile.",
        "analyze_error": "Your task is to analyze the error and provide a suggestion for fixing the error.",
        "verify_service_code": "Your task is to verify the service code and provide a corrected version of the code if the code is incorrect.",
        "llm_normalize": "You are a strict output normalizer for model inference evaluation.",
        "compose_request_body": "You are a file-to-field matcher. Given an input_schema and a listing of files in a test case's input/ directory (with content previews), you decide which file's content should fill each schema field. You are NOT making deployment decisions; you are a narrow assignment function.",
        "fix_input_schema": "You are an input schema specialist. Your job is to produce a corrected input_schema.json that is consistent with both the preprocess() code and the actual test input files.",
        "cross_function_alignment": "You are a post-fix reviewer. After a targeted fix to one function in an inference pipeline, you judge whether the full service.py is now acceptable or needs a full rewrite.",
    }
    return description_map[role]