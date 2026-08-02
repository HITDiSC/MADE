"""Standalone inference-result evaluation flow.

This module is a self-contained extraction of the *evaluation* pipeline that
lives inside ``agenttool/ServiceDelivery.py`` (the ``_normalize_with_llm`` ->
``_run_field_judges`` -> ``_compare_field_*`` chain and the pass/fail decision
in ``validate_inference_results``).

The logic here is a verbatim copy of the original — same LLM normalization
prompt, same field-judge thresholds, same lenient pass rule ("as long as the
normalizer could extract comparable fields, the format is consistent and the
case passes; scores are diagnostic only"), and the same observation/return
shapes. ServiceDelivery is left untouched; this file simply makes the same
flow importable on its own so it can be driven without spinning up Docker / a
running API — e.g. to re-score logged (expected_output, api_response_result)
pairs offline for ablation / log-replay.

Typical use:

    from agenttool.evaluation import InferenceEvaluator

    ev = InferenceEvaluator(backend="gr")
    comparison = ev.evaluate_case(
        expected_output=expected_str,
        api_response_result=actual_json_str,
        output_format_reference=output_format_md,   # optional
    )
    # comparison == {"passed": True, "fields": {...}}   (or no_comparable_fields)

    # Batch + pass/fail split + the same observations the tool returns:
    case_results = [ev.evaluate_case_entry(**c) for c in cases]
    observations = InferenceEvaluator.summarize(case_results)
"""

import os
import json
import difflib
from typing import Any, Dict, List

import yaml

from backend.query import json_query


# Resolve project paths exactly like ServiceDelivery does, so the prompt file
# is loaded from the same place.
_file_path = os.path.dirname(__file__)
_project_path = os.path.dirname(_file_path)
_prompt_path = os.path.join(_project_path, "prompts")


class InferenceEvaluator:
    """Encapsulates the LLM-normalize -> field-judge -> score evaluation flow.

    Parameters
    ----------
    backend:
        The query backend passed to ``json_query`` (defaults to "gr", matching
        ServiceDelivery's default).
    llm_normalize_prompt:
        Optional pre-loaded prompt dict. When omitted it is loaded from
        ``prompts/llm_normalize_prompt.json`` — the same file ServiceDelivery
        uses.
    """

    def __init__(self, backend: str = "gr", llm_normalize_prompt: Dict[str, Any] = None) -> None:
        self.backend = backend
        if llm_normalize_prompt is not None:
            self.llm_normalize_prompt = llm_normalize_prompt
        else:
            prompt_filepath = os.path.join(_prompt_path, "llm_normalize_prompt.json")
            with open(prompt_filepath, "r", encoding="utf-8") as f:
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
