from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PhaseSummarizer:
    """
    Summarize one phase execution into a compact PM-friendly structure.

    It still supports the old record API, but summarize(em_board) now produces
    a higher-signal phase summary that is easier to place into PM prompts.
    """

    def __init__(self, file_path: str | Path | None = None) -> None:
        self.file_path = str(file_path) if file_path is not None else None
        self.records: List[Dict[str, Any]] = []

    def set_file(self, file_path: str | Path) -> None:
        self.file_path = str(file_path)

    def add_record(
        self,
        summary: str,
        *,
        phase_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = {
            "file_path": self.file_path,
            "timestamp": timestamp or datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "phase_name": phase_name,
            "tool_name": tool_name,
            "status": status,
            "summary": summary,
            "details": details or {},
        }
        self.records.append(record)
        return record

    def add_error(
        self,
        summary: str,
        *,
        phase_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.add_record(
            summary,
            phase_name=phase_name,
            tool_name=tool_name,
            status="error",
            details=details,
        )

    def get_records(self) -> List[Dict[str, Any]]:
        return list(self.records)

    def clear(self) -> None:
        self.records.clear()

    def summarize(
        self,
        em_board: Optional[List[Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if em_board is None:
            latest_status = self.records[-1]["status"] if self.records else "idle"
            return {
                "file_path": self.file_path,
                "record_count": len(self.records),
                "latest_status": latest_status,
                "records": self.get_records(),
            }

        return self._summarize_em_board(em_board, observation or {})

    def _summarize_em_board(
        self,
        em_board: List[Any],
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        phase_name: Optional[str] = None
        successful_tools: List[str] = []
        # Error evidence for root-cause attribution. Previously errors were
        # dropped entirely, so PM could never tell WHAT failed - which made the
        # invalidate_phases / needs_redo decision (concluding an earlier
        # completed phase is actually broken) impossible from the summary alone.
        errors: List[Dict[str, str]] = []
        last_tool: Optional[str] = None
        # Track the phase's MOST RECENT outcome so we can tell a phase that hit
        # an error but RECOVERED (downloaded weights via a fallback, ended on a
        # successful tool) from one that is still broken. Only the latter should
        # drive root-cause attribution / invalidation - otherwise every phase
        # with a transient or expected error (e.g. find_local_weights "no local
        # weights, please download") would be wrongly flagged and re-run forever.
        last_outcome: Optional[str] = None  # "success" | "error"

        for item in em_board:
            if isinstance(item, dict):
                if "tool_name" in item:
                    last_tool = item.get("tool_name")
                elif "error" in item:
                    errors.append({
                        "tool": last_tool or "unknown",
                        "message": str(item["error"])[:400],
                    })
                    last_outcome = "error"
                continue

            if isinstance(item, str):
                text = item.strip()
                if not text:
                    continue

                if "phase:" in text:
                    phase_name = text.split("phase:", 1)[-1].strip()
                    continue

                if text.startswith("successfully run "):
                    tool_name = text.replace("successfully run ", "", 1).rstrip(".").strip()
                    successful_tools.append(tool_name)
                    last_outcome = "success"
                    continue

                if text.startswith("Variable ") and "successfully stored" in text:
                    last_outcome = "success"
                    continue

                low = text.lower()
                if low.startswith("error when running ") or low.startswith("error "):
                    body = text.split("running ", 1)[-1] if "running " in low else text
                    tool, sep, msg = body.partition(":")
                    errors.append({
                        "tool": (tool.strip().rstrip(".") if sep else (last_tool or "unknown")),
                        "message": (msg.strip() or body.strip())[:400],
                    })
                    last_outcome = "error"
                    continue

        successful_tools = self._dedupe_strings(successful_tools)
        errors = self._dedupe_errors(errors)[-5:]  # keep the most recent few
        had_errors = bool(errors)
        last_error = errors[-1] if errors else None
        ended_with_error = last_outcome == "error"
        # Attribution (and the invalidate signal it drives) fires ONLY when the
        # phase is still in a failed state at the end - not for recovered errors.
        failure_hint = self._root_cause_hint(errors) if ended_with_error else None

        if ended_with_error:
            status = "errors_present"
        elif successful_tools:
            status = "ok"  # may have had_errors=True but recovered
        else:
            status = "planned_only"

        summary_lines = [f"phase={phase_name or 'unknown'}", f"status={status}"]
        if successful_tools:
            summary_lines.append(f"successful={successful_tools}")
        if observation:
            summary_lines.append(f"observation_keys={sorted(observation.keys())}")
        if had_errors and not ended_with_error:
            summary_lines.append("note=had recoverable errors but recovered (last action succeeded)")
        if last_error and ended_with_error:
            summary_lines.append(f"last_error=({last_error['tool']}) {last_error['message']}")
        if failure_hint:
            summary_lines.append(f"root_cause_hint={failure_hint}")

        return {
            "phase": phase_name,
            "phase_name": phase_name,
            "status": status,
            "observation": observation,
            "successful_tools": successful_tools,
            "had_errors": had_errors,
            "ended_with_error": ended_with_error,
            "errors": errors,
            "last_error": last_error if ended_with_error else None,
            "failure_hint": failure_hint,
            "summary_for_pm": "; ".join(summary_lines),
        }

    def _dedupe_errors(self, errors: List[Dict[str, str]]) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        seen = set()
        for e in errors:
            key = (e.get("tool"), e.get("message"))
            if key in seen:
                continue
            seen.add(key)
            result.append(e)
        return result

    def _root_cause_hint(self, errors: List[Dict[str, str]]) -> Optional[str]:
        """Conservative keyword-based hint mapping a failure signature to the
        phase whose output is the likely root cause. Advisory only - PM still
        decides, and only invalidates an EARLIER, already-completed phase.
        """
        if not errors:
            return None
        text = " ".join(e.get("message", "").lower() for e in errors)
        cuda_kw = ("cuda", "no kernel image", "cudnn", "libcudnn", "nvidia",
                   "torch.cuda", "device-side assert", "gpu is not available")
        if any(k in text for k in cuda_kw):
            return ("CUDA/torch-runtime failure -> the docker environment is the likely "
                    "root cause; if dockersetup is already 'completed', consider "
                    "invalidate_phases=[dockersetup] and re-run it.")
        weight_kw = ("safetensors", "state_dict", "checkpoint", "no such file or directory",
                     "weight", ".bin", ".pt", "missing key", "size mismatch")
        if any(k in text for k in weight_kw):
            return ("missing/invalid model weights -> weightresolve is the likely root "
                    "cause; if weightresolve is already 'completed', consider "
                    "invalidate_phases=[weightresolve] and re-run it.")
        return None

    def _dedupe_strings(self, values: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result
