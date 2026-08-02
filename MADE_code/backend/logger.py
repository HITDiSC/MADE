"""Centralized logging for the autodeploy pipeline.

Tracks:
1. Token consumption (input / output / total) per LLM call and cumulative
2. Full input & output of every LLM call
3. log_p (logprobs) per LLM call when available
4. Wall-clock time of every LLM call, tool execution, and phase
5. Clear phase / role / step labels

Usage
-----
All LLM calls go through backend/query.py, which calls
``llm_logger.log_llm_call(...)`` after every request. Phase and tool
timing are handled in main.py via ``llm_logger.start_phase()`` /
``llm_logger.end_phase()`` and ``llm_logger.start_step()`` /
``llm_logger.end_step()``.

Logs are written to ``<project_root>/logs/<run_id>/`` as both a
machine-readable JSONL file and a human-readable summary.
"""

import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_project_root = Path(__file__).resolve().parent.parent


class PipelineLogger:
    """Thread-safe logger for a single pipeline run."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = _project_root / "logs" / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self.log_dir / "calls.jsonl"
        self._summary_path = self.log_dir / "summary.txt"
        self._lock = threading.Lock()

        # Cumulative token counters
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_llm_calls = 0
        self.total_llm_time = 0.0

        # Active phase / step context (per-thread)
        self._phase_ctx: Dict[int, Dict[str, Any]] = {}
        self._step_ctx: Dict[int, Dict[str, Any]] = {}

        self._write_summary_header()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tid(self) -> int:
        return threading.get_ident()

    def _append_jsonl(self, record: Dict[str, Any]):
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _append_summary(self, text: str):
        with self._lock:
            with open(self._summary_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def _write_summary_header(self):
        self._append_summary(f"{'='*80}")
        self._append_summary(f"  AutoDeploy Pipeline Log — Run {self.run_id}")
        self._append_summary(f"  Started: {datetime.now().isoformat()}")
        self._append_summary(f"{'='*80}\n")

    def _current_phase(self) -> str:
        ctx = self._phase_ctx.get(self._tid())
        return ctx["phase_name"] if ctx else "unknown"

    def _current_role(self) -> str:
        ctx = self._step_ctx.get(self._tid())
        return ctx["role"] if ctx else "unknown"

    # ------------------------------------------------------------------
    # Generic machine-readable event (coordination / ablation telemetry)
    # ------------------------------------------------------------------
    def log_event(self, event: str, **fields: Any):
        """Append an arbitrary structured event to calls.jsonl.

        Used for coordination telemetry (gate_decision, gate_stats) so that
        ablation policies can be reconstructed OFFLINE from a single run's log
        without re-executing the pipeline. Pure logging: no side effects on the
        pipeline. `phase` is auto-stamped from the current phase context.
        """
        record = {"event": event, "phase": self._current_phase()}
        record.update(fields)
        self._append_jsonl(record)

    # ------------------------------------------------------------------
    # Phase lifecycle
    # ------------------------------------------------------------------

    def start_phase(self, phase_name: str):
        tid = self._tid()
        self._phase_ctx[tid] = {
            "phase_name": phase_name,
            "start_time": time.time(),
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        header = (
            f"\n{'#'*80}\n"
            f"##  PHASE: {phase_name.upper()}\n"
            f"##  Started: {datetime.now().isoformat()}\n"
            f"{'#'*80}"
        )
        self._append_summary(header)

    def end_phase(self, phase_name: str):
        tid = self._tid()
        ctx = self._phase_ctx.pop(tid, None)
        if ctx is None:
            return
        elapsed = time.time() - ctx["start_time"]
        footer = (
            f"\n{'#'*80}\n"
            f"##  PHASE COMPLETE: {phase_name.upper()}\n"
            f"##  Duration: {elapsed:.2f}s | LLM calls: {ctx['llm_calls']} | "
            f"Tokens: {ctx['prompt_tokens']} in / {ctx['completion_tokens']} out\n"
            f"{'#'*80}\n"
        )
        self._append_summary(footer)
        self._append_jsonl({
            "event": "phase_end",
            "phase": phase_name,
            "duration_s": round(elapsed, 2),
            "llm_calls": ctx["llm_calls"],
            "prompt_tokens": ctx["prompt_tokens"],
            "completion_tokens": ctx["completion_tokens"],
        })

    # ------------------------------------------------------------------
    # Step lifecycle  (tool execution or EM/PM decision)
    # ------------------------------------------------------------------

    def start_step(self, role: str, step_name: str):
        tid = self._tid()
        self._step_ctx[tid] = {
            "role": role,
            "step_name": step_name,
            "start_time": time.time(),
        }
        self._append_summary(
            f"\n  ---- [{self._current_phase()}] STEP: {step_name} (role={role}) ----"
        )

    def end_step(self, step_name: str):
        tid = self._tid()
        ctx = self._step_ctx.pop(tid, None)
        if ctx is None:
            return
        elapsed = time.time() - ctx["start_time"]
        self._append_summary(
            f"  ---- STEP DONE: {step_name} | {elapsed:.2f}s ----"
        )
        self._append_jsonl({
            "event": "step_end",
            "phase": self._current_phase(),
            "role": ctx["role"],
            "step": step_name,
            "duration_s": round(elapsed, 2),
        })

    # ------------------------------------------------------------------
    # LLM call logging  (called from backend/query.py)
    # ------------------------------------------------------------------

    def log_llm_call(
        self,
        *,
        call_type: str,             # "query" | "json_query" | "code_query"
        role: str,                  # e.g. "em", "pm", "adapt_code"
        backend: str,               # "openai" | "gr"
        model: str,
        prompt: str,
        response_text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        logprobs: Optional[Any] = None,
        duration_s: float = 0.0,
    ):
        # Update cumulative counters
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.total_llm_calls += 1
            self.total_llm_time += duration_s

        # Update phase counters
        tid = self._tid()
        phase_ctx = self._phase_ctx.get(tid)
        if phase_ctx:
            phase_ctx["llm_calls"] += 1
            phase_ctx["prompt_tokens"] += prompt_tokens
            phase_ctx["completion_tokens"] += completion_tokens

        # JSONL record (machine-readable, includes full I/O)
        record = {
            "event": "llm_call",
            "timestamp": datetime.now().isoformat(),
            "phase": self._current_phase(),
            "role": role,
            "call_type": call_type,
            "backend": backend,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_s": round(duration_s, 3),
            "logprobs": logprobs,
            "prompt": prompt,
            "response": response_text,
        }
        self._append_jsonl(record)

        # Human-readable summary (truncated I/O)
        prompt_preview = prompt[:200].replace("\n", "\\n") + ("..." if len(prompt) > 200 else "")
        response_preview = response_text[:300].replace("\n", "\\n") + ("..." if len(response_text) > 300 else "")

        self._append_summary(
            f"\n    [LLM] {call_type} | role={role} | backend={backend} | model={model}\n"
            f"    Tokens: prompt={prompt_tokens} / completion={completion_tokens} / total={total_tokens}\n"
            f"    Time: {duration_s:.2f}s\n"
            f"    logprobs: {logprobs if logprobs else 'N/A'}\n"
            f"    Prompt:   {prompt_preview}\n"
            f"    Response: {response_preview}"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    def write_final_summary(self):
        summary = (
            f"\n{'='*80}\n"
            f"  PIPELINE RUN COMPLETE — {self.run_id}\n"
            f"  Finished: {datetime.now().isoformat()}\n"
            f"{'='*80}\n"
            f"  Total LLM calls:    {self.total_llm_calls}\n"
            f"  Total prompt tokens:     {self.total_prompt_tokens}\n"
            f"  Total completion tokens: {self.total_completion_tokens}\n"
            f"  Total tokens:        {self.total_tokens}\n"
            f"  Total LLM time:      {self.total_llm_time:.2f}s\n"
            f"{'='*80}\n"
        )
        self._append_summary(summary)
        self._append_jsonl({
            "event": "pipeline_end",
            "timestamp": datetime.now().isoformat(),
            "total_llm_calls": self.total_llm_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_llm_time_s": round(self.total_llm_time, 2),
        })


# Global singleton — initialized by main.py at startup, imported by query.py
llm_logger: Optional[PipelineLogger] = None


def init_logger(run_id: Optional[str] = None) -> PipelineLogger:
    global llm_logger
    llm_logger = PipelineLogger(run_id=run_id)
    return llm_logger


def get_logger() -> Optional[PipelineLogger]:
    return llm_logger
