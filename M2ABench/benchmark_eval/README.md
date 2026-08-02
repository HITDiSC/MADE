# Deployment Evaluator — Usage

```
evaluation.py               InferenceEvaluator + CLI
llm_normalize_prompt.json   prompt (auto-loaded from this folder)
```

## Library

```python
from evaluation import InferenceEvaluator

# Bring your own LLM caller (no MADE dependency):
ev = InferenceEvaluator(normalize_fn=my_normalizer)   # (prompt: str) -> dict
# or reuse MADE's backend:
ev = InferenceEvaluator(backend="gr")

result = ev.evaluate_case(expected_output, api_response_result)
# -> {"passed": True, "fields": {...}}  or  {"passed": False, "no_comparable_fields": True, ...}

# Batch: per-case pass/fail records
records = [ev.evaluate_case_entry(**c) for c in cases]
```

`normalize_fn` must return:

```json
{
  "normalized_expected": [{"field_name": "label", "field_value": "cat"}],
  "normalized_actual":   [{"field_name": "label", "field_value": "cat"}],
  "field_judges":        [{"field_name": "label", "field_judge_type": "exact"}]
}
```

## CLI

JSONL with `expected_output` / `api_response_result` per line:

```bash
python evaluation.py cases.jsonl --backend gr
python evaluation.py cases.jsonl --backend gr --out results.json
```
