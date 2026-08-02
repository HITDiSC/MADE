import json
import logging
import os
import time
from openai import OpenAI
import yaml
import requests
from pydantic import BaseModel
from agent.output_store import output_schema, role_description_text
from backend.logger import get_logger


# load global config
file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)
global_config = yaml.safe_load(open(os.path.join(project_path, "config/global.yaml"), "r"))

OPENAI_API_KEY = global_config.get("backend").get("openai_api_key")
KEY77_API_KEY = global_config.get("backend").get("key77_api_key")
GR_API_KEY = global_config.get("backend").get("gr_api_key")
# OpenAI-compatible endpoint for the "gr" backend. Read from config so no
# private endpoint is hard-coded; set backend.gr_base_url in config/global.yaml.
GR_BASE_URL = global_config.get("backend").get("gr_base_url") or "https://api.openai.com/v1"

# Number of top logprobs to request for the EM action call. 0 (default) means
# the parameter is NOT sent -> identical to prior behavior (zero risk if the
# endpoint doesn't support it). Set backend.em_top_logprobs > 0 to populate
# response logprobs so the `uncertainty` intervention policy can compute an
# EM-action uncertainty (offline from the log, or online once wired).
EM_TOP_LOGPROBS = int(global_config.get("backend", {}).get("em_top_logprobs", 0) or 0)


def _extract_usage(response) -> dict:
    """Extract token usage from an OpenAI response object.

    chat.completions.create -> CompletionUsage(prompt_tokens=22, completion_tokens=6, total_tokens=28)
    responses.parse         -> ResponseUsage(input_tokens=None, output_tokens=None, total_tokens=1141)
    """
    usage = response.usage
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or 0
    completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
    total = getattr(usage, "total_tokens", None) or 0
    # responses.parse may only return total_tokens; leave prompt/completion as 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _extract_logprobs(response) -> list:
    """Extract logprobs from a chat.completions.create response.

    response.choices[0].logprobs = ChoiceLogprobs(
        content=[ChatCompletionTokenLogprob(token='Paris', logprob=-0.2014, ...), ...])
    """
    try:
        lp = response.choices[0].logprobs
        if lp is not None and lp.content:
            return [t.logprob for t in lp.content]
    except Exception:
        pass
    return []


def _extract_logprobs_parsed(response) -> list:
    """Extract logprobs from a responses.parse response.

    response.output[0].content[0].logprobs = [LogProb(token=..., logprob=...), ...]
    Note: returns empty list [] by default; need top_logprobs > 0 to get values.
    """
    try:
        for item in response.output:
            for part in item.content:
                if part.logprobs:
                    return [t.logprob for t in part.logprobs]
    except Exception:
        pass
    return []


def _retry_on_rate_limit(func, max_retries=5, base_delay=10):
    """Wrapper that retries on rate-limit (429) errors with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "too many requests" in err_str or "429" in err_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"[query] rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                raise
    return wrapper


def _log(call_type, role, backend, model, prompt, response_text, usage, logprobs, duration):
    """Send a record to the pipeline logger if initialized."""
    logger = get_logger()
    if logger is None:
        return
    logger.log_llm_call(
        call_type=call_type,
        role=role,
        backend=backend,
        model=model,
        prompt=prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False, default=str),
        response_text=response_text if isinstance(response_text, str) else json.dumps(response_text, ensure_ascii=False, default=str),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        logprobs=logprobs or None,
        duration_s=duration,
    )


# =====================================================================
# Plain text query
# =====================================================================

def query_func(backend: str):
    backend_mapping = {
        "openai": query_openai,
        "gr": query_gr,
    }
    return backend_mapping[backend]

def query(prompt, backend: str) -> str:
    query_function = query_func(backend)
    response = _retry_on_rate_limit(query_function)(prompt)
    return response

def query_openai(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        logprobs=True,
    )
    duration = time.time() - t0
    text = response.choices[0].message.content
    _log("query", "general", "openai", "gpt-5.2", prompt, text,
         _extract_usage(response), _extract_logprobs(response), duration)
    return text

def query_gr(prompt):
    client = OpenAI(
        api_key=GR_API_KEY,
        base_url=GR_BASE_URL
    )
    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        logprobs=True,
    )
    duration = time.time() - t0
    text = response.choices[0].message.content
    _log("query", "general", "gr", "gpt-5.2", prompt, text,
         _extract_usage(response), _extract_logprobs(response), duration)
    return text


# =====================================================================
# JSON structured query
# =====================================================================

def _logprobs_kwargs(role: str) -> dict:
    """top_logprobs kwargs for responses.parse, only for the EM action call.

    Returns {} unless backend.em_top_logprobs > 0 AND this is the EM action
    ("em") -- so by default nothing changes and no unsupported parameter is ever
    sent. Restricted to role=="em" because only the EM action feeds the
    uncertainty policy; PM/summarizer calls need no logprobs.
    """
    if EM_TOP_LOGPROBS > 0 and role == "em":
        return {"top_logprobs": EM_TOP_LOGPROBS}
    return {}


def json_query_func(backend: str):
    backend_mapping = {
        "openai": json_query_openai,
        "gr": json_query_gr,
    }
    return backend_mapping[backend]

def json_query(prompt, role, backend: str) -> str:
    query_function = json_query_func(backend)
    response = _retry_on_rate_limit(query_function)(prompt, role)
    return response

def json_query_openai(prompt, role):
    role_schema = output_schema(role)
    role_description = role_description_text(role)
    client = OpenAI(api_key=OPENAI_API_KEY)
    t0 = time.time()
    parse_kwargs = _logprobs_kwargs(role)
    response = client.responses.parse(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": role_description},
            {"role": "user", "content": prompt},
        ],
        text_format=role_schema,
        **parse_kwargs,
    )
    duration = time.time() - t0
    response_string = response.output_parsed
    response_json = json.loads(response_string.model_dump_json())
    _log("json_query", role, "openai", "gpt-5.2", prompt, response_json,
         _extract_usage(response), _extract_logprobs_parsed(response), duration)
    return response_json

def json_query_gr(prompt, role):
    role_schema = output_schema(role)
    role_description = role_description_text(role)
    client = OpenAI(
        api_key=GR_API_KEY,
        base_url=GR_BASE_URL
    )
    t0 = time.time()
    parse_kwargs = _logprobs_kwargs(role)
    response = client.responses.parse(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": role_description},
            {"role": "user", "content": prompt},
        ],
        text_format=role_schema,
        **parse_kwargs,
    )
    duration = time.time() - t0
    response_string = response.output_parsed
    response_json = json.loads(response_string.model_dump_json())
    _log("json_query", role, "gr", "gpt-5.2", prompt, response_json,
         _extract_usage(response), _extract_logprobs_parsed(response), duration)
    return response_json


# =====================================================================
# Code generation query
# =====================================================================

def code_query(prompt, role, backend: str):
    backend_mapping = {
        "openai": code_query_openai,
        "gr": code_query_gr,
    }
    query_function = backend_mapping[backend]
    response = _retry_on_rate_limit(query_function)(prompt, role)
    return response

def code_query_openai(prompt, role):
    role_schema = output_schema(role)
    role_description = role_description_text(role)
    client = OpenAI(api_key=OPENAI_API_KEY)
    t0 = time.time()
    response = client.responses.parse(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": role_description},
            {"role": "user", "content": prompt},
        ],
        text_format=role_schema,
    )
    duration = time.time() - t0
    response_string = response.output_parsed
    response_json = json.loads(response_string.model_dump_json())
    _log("code_query", role, "openai", "gpt-5.2", prompt, response_json,
         _extract_usage(response), _extract_logprobs_parsed(response), duration)
    return response_json

def code_query_gr(prompt, role):
    role_schema = output_schema(role)
    role_description = role_description_text(role)
    client = OpenAI(
        api_key=GR_API_KEY,
        base_url=GR_BASE_URL
    )
    t0 = time.time()
    response = client.responses.parse(
        model="gpt-5.2",
        input=[
            {"role": "system", "content": role_description},
            {"role": "user", "content": prompt},
        ],
        text_format=role_schema,
    )
    duration = time.time() - t0
    response_string = response.output_parsed
    response_json = json.loads(response_string.model_dump_json())
    _log("code_query", role, "gr", "gpt-5.2", prompt, response_json,
         _extract_usage(response), _extract_logprobs_parsed(response), duration)
    return response_json
