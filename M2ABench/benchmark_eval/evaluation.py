"""Standalone inference-result evaluation flow (benchmark-portable).

Extracted from MADE's ServiceDelivery (the ``_normalize_with_llm`` ->
``_run_field_judges`` -> ``_compare_field_*`` chain and the pass/fail decision
in ``validate_inference_results``). The *scoring logic is unchanged* — same LLM
normalization prompt, same field-judge thresholds, same lenient pass rule ("as
long as the normalizer could extract comparable fields, the format is
consistent and the case passes; scores are diagnostic only"), same return
shapes.

Only the plumbing is made portable so this file can live in a benchmark folder
by itself:

  * The LLM normalizer is injectable. Pass ``normalize_fn`` — any callable
    ``(prompt_str) -> dict`` that returns the JSON the ``llm_normalize`` prompt
    asks for. If omitted, it lazily imports MADE's ``backend.query.json_query``
    at call time (so merely importing this module never fails when ``backend``
    is absent).
  * The prompt is flexible: pass a dict (``llm_normalize_prompt=``), a path
    (``prompt_path=``), or drop ``llm_normalize_prompt.json`` next to this file
    and it is found automatically.

Usage as a library:

    from evaluation import InferenceEvaluator

    # A) with your own LLM caller (recommended in a benchmark):
    def my_normalizer(prompt: str) -> dict:
        ...  # call any OpenAI-compatible model, return parsed JSON
    ev = InferenceEvaluator(normalize_fn=my_normalizer)

    # B) reusing MADE's backend (when this sits inside the MADE tree):
    ev = InferenceEvaluator(backend="gr")

    comparison = ev.evaluate_case(expected_output, api_response_result,
                                  output_format_reference="")   # optional
    # -> {"passed": True, "fields": {...}}   or   {"passed": False, "no_comparable_fields": True, ...}

    case_results = [ev.evaluate_case_entry(**c) for c in cases]
    observations = InferenceEvaluator.summarize(case_results)

Usage as a CLI (batch re-scoring of a JSONL file, one object per line with
keys ``expected_output`` / ``api_response_result`` and optional
``output_format_reference`` / ``case_name``); needs MADE's backend on the path
unless you wire your own normalizer in code:

    python evaluation.py cases.jsonl --backend gr
"""

import os
import json
import difflib
from typing import Any, Callable, Dict, List, Optional


def _default_prompt_path() -> Optional[str]:
    """Find llm_normalize_prompt.json: next to this file, then ../prompts/."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "llm_normalize_prompt.json"),
        os.path.join(os.path.dirname(here), "prompts", "llm_normalize_prompt.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


class InferenceEvaluator:
    """Encapsulates the LLM-normalize -> field-judge -> score evaluation flow.

    Parameters
    ----------
    backend:
        Backend name forwarded to the default MADE normalizer
        (``json_query(prompt, "llm_normalize", backend)``). Ignored when
        ``normalize_fn`` is supplied.
    normalize_fn:
        Optional ``(prompt_str) -> dict`` callable that performs the LLM
        normalization. When given, this module has no dependency on MADE's
        backend. When ``None``, ``backend.query.json_query`` is imported lazily
        on first use.
    llm_normalize_prompt:
        Optional pre-loaded prompt dict. Takes precedence over ``prompt_path``.
    prompt_path:
        Optional path to ``llm_normalize_prompt.json``. When both this and
        ``llm_normalize_prompt`` are omitted, the file is looked up next to this
        module (and then in ``../prompts/``).
    """

    def __init__(
        self,
        backend: str = "gr",
        normalize_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        llm_normalize_prompt: Optional[Dict[str, Any]] = None,
        prompt_path: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self._normalize_fn = normalize_fn

        if llm_normalize_prompt is not None:
            self.llm_normalize_prompt = llm_normalize_prompt
        else:
            path = prompt_path or _default_prompt_path()
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(
                    "llm_normalize_prompt.json not found. Place it next to "
                    "evaluation.py, pass prompt_path=..., or pass "
                    "llm_normalize_prompt=<dict>."
                )
            with open(path, "r", encoding="utf-8") as f:
                self.llm_normalize_prompt = json.load(f)

    # ------------------------------------------------------------------
    # pure scoring helpers (deterministic, no LLM)
    # ------------------------------------------------------------------

    def _normalize_text(self, value: Any) -> str:
        return " ".join(str(value).strip().lower().split())

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

    # ------------------------------------------------------------------
    # field-judge evaluation (pass/fail decision + per-field diagnostics)
    # ------------------------------------------------------------------

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

    def _run_field_judges(self, normalization: Dict[str, Any]) -> Dict[str, Any]:
        return self._evaluate_llm_judge_result(normalization)

    # ------------------------------------------------------------------
    # LLM normalization
    # ------------------------------------------------------------------

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

    def _call_normalizer(self, prompt: str) -> Dict[str, Any]:
        """Invoke the injected normalizer, or lazily fall back to MADE's
        ``backend.query.json_query`` so importing this module never requires the
        MADE backend to be present."""
        if self._normalize_fn is not None:
            return self._normalize_fn(prompt)
        from backend.query import json_query  # lazy: only needed at call time
        return json_query(prompt, "llm_normalize", self.backend)

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
        response = self._call_normalizer(prompt)
        if not isinstance(response, dict):
            raise RuntimeError("LLM normalization response must be a JSON object")
        return response

    # ------------------------------------------------------------------
    # top-level orchestration (mirrors the per-case core of
    # validate_inference_results, minus the Docker / API-inference parts)
    # ------------------------------------------------------------------

    def evaluate_case(
        self,
        expected_output: str,
        api_response_result: str,
        output_format_reference: str = "",
    ) -> Dict[str, Any]:
        """Normalize -> judge one (expected, actual) pair. Returns the
        comparison dict: ``{"passed": True, "fields": {...}}`` on a
        format-consistent case, or ``{"passed": False, "no_comparable_fields":
        True, "reason": ...}`` when nothing comparable could be extracted.

        Note: the original passes ``actual_result=expected_output`` into the
        normalizer (the "actual result" reference block is the *expected*
        output, while ``api_response_result`` is the real model response). That
        ordering is preserved here.
        """
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
        return comparison

    def evaluate_case_entry(
        self,
        expected_output: str,
        api_response_result: str,
        output_format_reference: str = "",
        case_name: str = "",
        request_inputs: Any = None,
    ) -> Dict[str, Any]:
        """Same as :meth:`evaluate_case` but returns the full per-case record
        (``case_name`` / ``request_inputs`` / ``expected_output`` /
        ``api_response_result`` + the comparison fields spread in), matching the
        ``case_entry`` dict that ``validate_inference_results`` accumulates.
        """
        comparison = self.evaluate_case(
            expected_output=expected_output,
            api_response_result=api_response_result,
            output_format_reference=output_format_reference,
        )
        return {
            "case_name": case_name,
            "request_inputs": request_inputs,
            "expected_output": expected_output,
            "api_response_result": api_response_result,
            **comparison,
        }

    @staticmethod
    def summarize(case_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reproduce the observations returned by ``validate_inference_results``
        given a list of per-case records: ``validation_results`` +
        ``validation_failures``, plus ``validation_correctness_error`` only when
        *every* case failed (the lenient "≥1 case passes = success" rule).
        """
        if not case_results:
            raise RuntimeError(
                "No validation cases supplied. Expected at least one per-case "
                "record from evaluate_case_entry()."
            )

        passed_cases = [item for item in case_results if item.get("passed")]
        failed_cases = [item for item in case_results if not item.get("passed")]

        observations = [
            {
                "value": case_results,
                "storage": "temporary",
                "variable_name": "validation_results",
            },
            {
                "value": failed_cases,
                "storage": "temporary",
                "variable_name": "validation_failures",
            },
        ]
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
                "variable_name": "validation_correctness_error",
            })
        return observations


# ----------------------------------------------------------------------
# CLI: batch re-score a JSONL of cases.
# Each line: {"expected_output": ..., "api_response_result": ...,
#             "output_format_reference": "" (opt), "case_name": "" (opt)}
# ----------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Re-score inference cases from a JSONL file.")
    parser.add_argument("cases_jsonl", help="Path to JSONL; one case object per line.")
    parser.add_argument("--backend", default="gr", help="Backend for MADE's json_query (default: gr).")
    parser.add_argument("--prompt", default=None, help="Path to llm_normalize_prompt.json (optional).")
    parser.add_argument("--out", default=None, help="Write per-case results JSON here (optional).")
    args = parser.parse_args(argv)

    ev = InferenceEvaluator(backend=args.backend, prompt_path=args.prompt)

    cases = []
    with open(args.cases_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    case_results = [ev.evaluate_case_entry(**c) for c in cases]
    passed = sum(1 for r in case_results if r.get("passed"))
    print(f"cases: {len(case_results)}  passed: {passed}  failed: {len(case_results) - passed}")
    for r in case_results:
        print(f"  [{'PASS' if r.get('passed') else 'FAIL'}] {r.get('case_name') or '(unnamed)'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(case_results, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
