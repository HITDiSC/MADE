"""FastAPI template app.py.

Copied into the target repo by APIAdaptation.establish_fastapi_app. Exposes
two prediction endpoints:

    POST /predict_text   - new JSON-body contract, reads .autodeploy/input_schema.json
                           on every request and validates the body against it.
                           Passes the parsed `inputs` dict to predict() as
                           raw_input, which preprocess() then unpacks via
                           raw_input['<field_name>'] per the schema.

    POST /predict        - legacy single-file multipart contract. Preserved so
                           existing test harnesses (single file -> POST) keep
                           working. It reads the schema, figures out the single
                           string field name, decodes the uploaded file as utf-8
                           text, and routes the call through the same predict()
                           with a one-field dict as raw_input.

Input schema is deliberately re-read on every request (no caching) so that a
ServiceDelivery.fix_runtime_code call that rewrites .autodeploy/input_schema.json
takes effect without restarting uvicorn. File IO cost is negligible next to
inference latency.
"""

from fastapi import FastAPI, UploadFile, File, Body
import json
import os
import pickle
import traceback
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service import predict, load_model

app = FastAPI()
model_bundle = None

# Assume repo_root is mounted one level above this file inside the container
# (fastapiapp/ is a subdirectory of repo_root). If the mount layout differs
# we fall back to None and the endpoints degrade to "no schema available".
_REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = _REPO_ROOT / ".autodeploy" / "input_schema.json"


def _load_schema():
    """Return the current input_schema dict, or None if the file is missing/invalid.

    Re-read on every request intentionally (see module docstring).
    """
    if not SCHEMA_PATH.exists():
        return None
    try:
        return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _validate_inputs_against_schema(schema, inputs):
    """Return None if inputs satisfies schema.inputs, else a human-readable error string.

    Schema uses the LIST form: {"inputs": [{"field_name", "field_type",
    "field_required", "field_description"}, ...]}. There is no `params`.

    All input values are expected to be file-path strings, so type checking
    only verifies that required fields are present and that no unexpected
    fields are sent. The logical field_type is informational only.
    """
    declared_list = []
    if isinstance(schema, dict):
        raw_inputs = schema.get("inputs") or []
        if isinstance(raw_inputs, list):
            declared_list = [item for item in raw_inputs if isinstance(item, dict)]

    declared_names = set()

    for item in declared_list:
        name = item.get("field_name")
        if not isinstance(name, str) or not name:
            continue
        declared_names.add(name)

        required = bool(item.get("field_required", False))
        if required and name not in inputs:
            return f"missing required field '{name}'"

        # Validate that the file path exists inside the container
        if name in inputs:
            value = inputs[name]
            if isinstance(value, str) and value.startswith("/"):
                if not Path(value).exists():
                    return f"field '{name}' points to non-existent file: {value}"

    unexpected = set(inputs) - declared_names
    if unexpected:
        return f"unexpected fields not in schema: {sorted(unexpected)}"

    return None


@app.on_event("startup")
def startup_event():
    global model_bundle
    model_bundle = load_model()


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/predict_text")
async def run_prediction_text(payload: dict = Body(...)):
    """New JSON-body contract.

    Expected body:
        {
          "inputs": {"<field_name>": <file_path>, ...},
          "params": {...}    # optional
        }

    Each value in `inputs` is a file path inside the container. The paths
    are passed through to predict() as-is — preprocess() is responsible for
    reading the files from disk.
    """
    inputs = payload.get("inputs", {}) or {}
    params = payload.get("params", {}) or {}

    if not isinstance(inputs, dict):
        return {"success": False, "error": "payload.inputs must be a JSON object"}

    schema = _load_schema()

    if schema is not None:
        err = _validate_inputs_against_schema(schema, inputs)
        if err is not None:
            return {"success": False, "error": f"schema mismatch: {err}"}

    try:
        # predict(raw_input, model_bundle) - preprocess inside will do
        # raw_input['<field>'] lookups based on the schema it declared.
        result = predict(inputs, model_bundle)

        # Persist the result like the legacy endpoint does, for audit / replay.
        os.makedirs("result", exist_ok=True)
        filename = f"{uuid.uuid4().hex}.pkl"
        with open(os.path.join("result", filename), "wb") as f:
            pickle.dump(result, f)

        return {"success": True, "result": result}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


@app.post("/predict")
async def run_prediction_legacy(file: UploadFile = File(...)):
    """Legacy multipart single-file endpoint.

    Kept alive so the current test harness (_post_test_file sends one file as
    multipart) keeps working without any change. The translation rule is:

        - Read the current schema.
        - If the schema has exactly ONE required string field, decode the
          uploaded bytes as utf-8 and wrap them as {<that_field>: text} before
          calling predict(), giving the same predict() signature both endpoints
          share.
        - Otherwise (no schema, or multi-field schema), refuse with a clear
          error message telling the caller to use /predict_text with a JSON
          body describing each field.
    """
    file_bytes = await file.read()
    schema = _load_schema()

    if schema is None:
        # Pre-schema era fallback: pass raw bytes through as-is and let the
        # legacy preprocess() figure it out. This matches the pre-refactor
        # behavior for repos adapted before input_schema existed.
        try:
            result = predict(file_bytes, model_bundle)
            return {"success": True, "result": result}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    declared_inputs = schema.get("inputs") or []
    if not isinstance(declared_inputs, list):
        declared_inputs = []
    required_string_fields = [
        item["field_name"] for item in declared_inputs
        if isinstance(item, dict)
        and isinstance(item.get("field_name"), str)
        and item.get("field_type") == "string"
        and item.get("field_required") is True
    ]
    if len(required_string_fields) != 1:
        return {
            "success": False,
            "error": (
                f"legacy /predict endpoint only supports schemas with exactly one "
                f"required string field; current schema has {len(required_string_fields)} "
                f"required string fields: {required_string_fields}. "
                f"Use POST /predict_text with a JSON body instead."
            ),
        }

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "success": False,
            "error": "uploaded file is not valid utf-8 and schema expects a string field",
        }

    inputs = {required_string_fields[0]: text}
    try:
        result = predict(inputs, model_bundle)

        os.makedirs("result", exist_ok=True)
        filename = f"{uuid.uuid4().hex}.pkl"
        with open(os.path.join("result", filename), "wb") as f:
            pickle.dump(result, f)

        return {"success": True, "result": result}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"{type(e).__name__}: {e}"}
