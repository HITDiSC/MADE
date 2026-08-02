import argparse
from ast import arguments
import json
import multiprocessing
import threading
import time
import os
import sys
import yaml
import tiktoken
import difflib
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED

from agenttool.base_phase import BasePhase
from agent.phase_manager import PhaseManager
from agent.excution_master import ExecutionMaster
from agent.parallel_discriminator import ParallelDiscriminator
from agenttool.RepoIngest import RepoIngest
from agenttool.WeightResolve import WeightResolve
from agenttool.DockerSetUp import DockerSetUp
from agenttool.tool import *
from agenttool.APIAdaptation import APIAdaptation
from agenttool.ServiceDelivery import ServiceDelivery
from agenttool.PhaseSummarizer import PhaseSummarizer
from backend.logger import init_logger, get_logger
from agent.coordination import (
    consistency_gate, attribute_root_cause, known_artifacts, new_gate_stats,
    artifact_status,
)



file_path = os.path.dirname(__file__)
project_path = os.path.dirname(file_path)
global_config = yaml.safe_load(open(os.path.join(file_path, "config/global.yaml"), "r"))
max_phase_turns = global_config.get("agent").get("max_phase_turns")
max_execution_turns = global_config.get("agent").get("max_execution_turns")
max_query_retry_times = global_config.get("agent").get("max_query_retry_times")
max_token = global_config.get("agent").get("max_token_length")
enable_negotiation = global_config.get("agent").get("enable_negotiation", False)
max_negotiation_rounds = global_config.get("agent").get("max_negotiation_rounds", 2)
enable_gap_trigger = global_config.get("agent").get("enable_gap_trigger", False)
gap_repeat_threshold = global_config.get("agent").get("gap_repeat_threshold", 3)
gap_error_threshold = global_config.get("agent").get("gap_error_threshold", 2)
gap_max_interventions = global_config.get("agent").get("gap_max_interventions", 3)
gap_cooldown = global_config.get("agent").get("gap_cooldown", 2)
# Coordination mechanism (consistency-gated supervision) config + ablation switch.
intervention_policy = global_config.get("agent").get("intervention_policy", "consistency")
enable_override = global_config.get("agent").get("enable_override", True)
enable_retroactive_invalidation = global_config.get("agent").get("enable_retroactive_invalidation", True)
random_gate_p = global_config.get("agent").get("random_gate_p", 0.15)
ablation_seed = global_config.get("agent").get("ablation_seed", 0)
_gate_rng = random.Random(ablation_seed)

PHASE = {
    "repoingest": RepoIngest,
    "weightresolve": WeightResolve,
    "dockersetup": DockerSetUp,
    "apiadaptation": APIAdaptation,
    "servicedelivery": ServiceDelivery,
}

PHASE_LOOKUP = {
    name.lower(): name
    for name in PHASE
}

allowed_phases = ["repoingest", "weightresolve", "dockersetup", "apiadaptation", "servicedelivery"]
allowed_parallel_phases = ["weightresolve", "dockersetup"]

# Deterministic pipeline order for the non-monotonic Update operator: a failure
# may retroactively invalidate ONLY an EARLIER, already-executed phase (you
# cannot invalidate a phase that has not run yet). weightresolve/dockersetup run
# in parallel; their list index is used as the order. Prevents the false
# positive where e.g. a git_clone network error in repoingest "invalidates" the
# not-yet-run weightresolve.
PHASE_ORDER = {name: i for i, name in enumerate(allowed_phases)}


def _phase_precedes(candidate: str, current: str) -> bool:
    """True iff `candidate` runs strictly before `current` in the pipeline."""
    return (PHASE_ORDER.get(correct_phase_name(candidate), 10**9)
            < PHASE_ORDER.get(correct_phase_name(current), -1))

variable_store = VariableStore()

# Phases that PM has flagged as defective (outputs present but proven broken by a
# downstream failure). Persists across PM turns so a phase that merely "looks
# completed" by variable presence is shown as needs_redo until it is actually
# re-run. Cleared when that phase is re-executed. Lock-guarded for the parallel
# (ThreadPoolExecutor) phase block.
invalidated_phases: set = set()
invalidated_phases_lock = threading.Lock()


def mark_phases_invalid(phase_names):
    if not phase_names:
        return
    with invalidated_phases_lock:
        for p in phase_names:
            invalidated_phases.add(correct_phase_name(p))


def clear_phase_invalidation(phase_name):
    with invalidated_phases_lock:
        invalidated_phases.discard(correct_phase_name(phase_name))


def snapshot_invalidated():
    with invalidated_phases_lock:
        return set(invalidated_phases)


# Checkpoint file name written to logs/<run_id>/ right after the
# (weightresolve + dockersetup) parallel block finishes. Resume runs
# load this file via --resume_from to skip repoingest/weightresolve/
# dockersetup and go straight into apiadaptation.
PARALLEL_CHECKPOINT_FILENAME = "variable_store_after_parallel.json"


def _save_parallel_checkpoint(
    log_dir: Path,
    variable_store_: "VariableStore",
    pm_board_: list,
    phase_name_list_: list,
    phase_change_turns_: int,
) -> Path:
    """Dump state after the parallel phases so downstream phases can be
    tested in isolation on a later run via --resume_from.
    """
    path = Path(log_dir) / PARALLEL_CHECKPOINT_FILENAME
    payload = {
        "variable_store": variable_store_.get_all(),
        "pm_board": pm_board_,
        "phase_name_list": phase_name_list_,
        "phase_change_turns": phase_change_turns_,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            # default=str stringifies Path / non-JSON values; downstream
            # phases wrap what they receive with Path(...) anyway.
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"[main] parallel checkpoint saved -> {path}")
    except Exception as e:
        print(f"[main] WARNING: failed to save parallel checkpoint: {e}")
    return path


def _load_parallel_checkpoint(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_variable_token_size(variable_value, model="gpt-4o"):
    if isinstance(variable_value, str):
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(variable_value)
        return len(tokens)
    elif isinstance(variable_value, dict):
        return sum(get_variable_token_size(value) for value in variable_value.values())
    elif isinstance(variable_value, list):
        return sum(get_variable_token_size(item) for item in variable_value)
    else:
        return 0

def judge_store_variable(variable_value: Any):
    if get_variable_token_size(variable_value) > max_token:
        return False
    return True

MAX_SINGLE_ENTRY_CHARS = 3000   # cap for any single board entry
MAX_BOARD_CHARS = 800000        # ~200k tokens; compress old history when exceeded

def _truncate_entry(entry, limit=MAX_SINGLE_ENTRY_CHARS):
    """Truncate a single board entry (str or dict) to *limit* characters."""
    if isinstance(entry, str):
        if len(entry) > limit:
            return entry[:limit] + f"\n...[truncated, total {len(entry)} chars]"
        return entry
    if isinstance(entry, dict):
        result = {}
        for k, v in entry.items():
            sv = str(v)
            if len(sv) > limit:
                result[k] = sv[:limit] + f"\n...[truncated, total {len(sv)} chars]"
            else:
                result[k] = v
        return result
    sv = str(entry)
    if len(sv) > limit:
        return sv[:limit] + f"\n...[truncated, total {len(sv)} chars]"
    return entry

def _board_char_size(board):
    return sum(len(str(item)) for item in board)

def _compress_board(em_board, arguments_board, phase_summarizer, keep_recent=6):
    """Compress old em_board entries into a summary, keep recent ones verbatim."""
    if len(em_board) <= keep_recent:
        return em_board, arguments_board
    old_part = em_board[:-keep_recent]
    recent_part = em_board[-keep_recent:]
    summary = phase_summarizer.summarize(old_part, observation={})
    compressed = [
        f"[compressed history] {summary.get('summary_for_pm', str(summary))}"
    ]
    # Also trim arguments_board: keep only the first entry (initial variables)
    # and entries added during recent turns
    trimmed_args = arguments_board[:1] if arguments_board else []
    return compressed + recent_part, trimmed_args


def correct_phase_name(name: str, cutoff: float = 0.75) -> str:
    name = name.strip().lower()
    matches = difflib.get_close_matches(name, allowed_phases, n=1, cutoff=cutoff)
    return matches[0] if matches else name


def compute_deployment_state(available_phases, all_vars, invalidated=None):
    """Deterministic belief state (Z_hat): for each phase, whether its declared
    required_outputs are already present in the shared variable store.

    This is the persistent-belief realization of the paper's Ẑ^{t+1}=Update(Ẑ,o):
    EM's produced permanent variables ARE o^t, and the variable store accumulates
    them, so the belief updates for free with no extra LLM call. It is fed to PM
    on every decision so PM does not route control back to an already-completed
    phase based on a single fresh per-call summary (which can misread done work
    as undone). Status per phase: completed | partial | not_started.

    `invalidated` is the set of phases PM has flagged as defective: their outputs
    are present but proven broken by a downstream failure (variable presence does
    NOT prove correctness). Such phases are reported as 'needs_redo' so PM re-runs
    them instead of trusting the stale outputs.
    """
    invalidated = invalidated or set()
    state = []
    for ph in available_phases or []:
        name = ph.get("name")
        req = ph.get("required_outputs", []) or []
        present = [r for r in req
                   if r in all_vars and all_vars[r] not in (None, "", [], {})]
        missing = [r for r in req if r not in present]
        if name in invalidated:
            status = "needs_redo"
        elif req and not missing:
            status = "completed"
        elif present:
            status = "partial"
        else:
            status = "not_started"
        entry = {
            "phase": name,
            "status": status,
            "present_outputs": present,
            "missing_outputs": missing,
        }
        if status == "needs_redo":
            entry["note"] = "outputs present but flagged defective by a downstream failure; MUST be re-run (do not trust)"
        state.append(entry)
    return state


# --- Information-gap trigger (Component 2): cheap, em_board-derived signals that
# EM may be stuck, used to decide whether PM should intervene mid-phase. Derived
# from em_board rather than parallel counters so the delicate EM loop (with its
# many early `continue`s) does not need new bookkeeping at every branch.

def _entry_is_em_decision(e):
    return isinstance(e, dict) and "tool_name" in e


def _entry_is_error(e):
    if isinstance(e, dict):
        return "error" in e
    if isinstance(e, str):
        return e.startswith("error when running") or e.startswith("error ")
    return False


def _recent_tool_names(em_board, n):
    tools = [e["tool_name"] for e in em_board if _entry_is_em_decision(e)]
    return tools[-n:]


def _trailing_error_turns(em_board):
    """Count, from the end, how many consecutive EM turns ended in an error.

    A 'turn' spans one em_decision entry up to the next; the turn errored if any
    error marker appears in that span.
    """
    idxs = [i for i, e in enumerate(em_board) if _entry_is_em_decision(e)]
    if not idxs:
        return 0
    count = 0
    for j in range(len(idxs) - 1, -1, -1):
        start = idxs[j]
        end = idxs[j + 1] if j + 1 < len(idxs) else len(em_board)
        if any(_entry_is_error(e) for e in em_board[start:end]):
            count += 1
        else:
            break
    return count


def gap_signal_fired(em_board, repeat_threshold, error_threshold):
    """True if EM looks stuck: same tool repeat_threshold turns in a row, OR
    error_threshold consecutive errored turns."""
    recent = _recent_tool_names(em_board, repeat_threshold)
    repeated = len(recent) == repeat_threshold and len(set(recent)) == 1
    errored = _trailing_error_turns(em_board) >= error_threshold
    return repeated or errored

def execute_phase(phase_name: str, phase_manager: PhaseManager, backend: str, pm_board: list[dict], phase_name_list: list[str]):
    print("execute_phase", phase_name)
    phase_name = correct_phase_name(phase_name)
    # We are (re-)running this phase now, so any prior "defective" flag on it is
    # being addressed - clear it before this run produces fresh outputs.
    clear_phase_invalidation(phase_name)
    phase_summarizer = PhaseSummarizer()

    logger = get_logger()
    if logger:
        logger.start_phase(phase_name)

    phase = PHASE[phase_name]
    phase = phase()

    boundary_tool_used = False
    tmp_variable_store = VariableStore() # temporary variable store
    em_board = []
    gate_stats = new_gate_stats()  # per-phase mechanism statistics (predicate fires, decisions)
    next_phase = []
    phase_observation: dict = {} # permanent artifacts emitted during this phase
    # Last decision returned by phase_manager for this phase. None means PM
    # was never consulted (e.g. inner loop only ever called read_store_variable
    # and ran out of execution turns). main() uses this to decide whether
    # servicedelivery actually finalized the deployment or whether it needs a
    # retry / a fix-up phase.
    final_decision = None
    last_pm_decision = None

    arguments_board = []
    arguments_board.append(variable_store.get_all())


    em_board.append(f"You are now in the phase:{phase_name}")

    em_called_arguments = []

    execution_master = ExecutionMaster(backend=backend)
    execution_turns = 0
    gap_interventions = 0          # mid-phase PM nudges issued this phase
    last_gap_turn = -10**9         # turn index of the last mid-phase nudge
    while execution_turns < max_execution_turns: # run em
        execution_turns += 1
        # boundary_tool_used = False

        print(phase_name, "execution_turns", execution_turns)

        # Compress boards if accumulated size exceeds limit
        total_size = _board_char_size(em_board) + _board_char_size(arguments_board)
        if total_size > MAX_BOARD_CHARS:
            print(f"[{phase_name}] board size {total_size} exceeds {MAX_BOARD_CHARS}, compressing...")
            em_board, arguments_board = _compress_board(em_board, arguments_board, phase_summarizer)

        # NOTE: the mid-phase intervention gate was RELOCATED to just after EM
        # produces its proposed action (below), because the consistency
        # predicates (redundant / premature / stale) inspect the PROPOSED tool +
        # args, which do not exist yet at the top of the loop. Only P_stall is
        # history-derived; it is now one predicate among four inside
        # consistency_gate().

        if logger:
            logger.start_step("execution_master", f"em_turn_{execution_turns}")
        em_response = execution_master.run(phase_description=phase.description, goal=phase.goal, board = em_board, tool_list=phase.tools_schemas, arguments_board=arguments_board+em_called_arguments)
        if logger:
            logger.end_step(f"em_turn_{execution_turns}")
        print(phase_name, "em_response", em_response)

        em_decision = em_response
        em_board.append(em_decision)

        tool_name = em_decision["tool_name"]

        if phase.boundary_tools(tool_name): # decide whether pm should take actions to stop the phase
            boundary_tool_used = True

        em_called_arguments = []

        em_arguments = em_decision["arguments"]

        # Validate EM's chosen tool before dispatching. EM's prompt allows it to
        # "indicate no tool", and it can also hallucinate a wrong name; either
        # way phase.tool_arguments / phase.tools would KeyError and crash the
        # phase. Instead of crashing (or silently looping), fold it into the
        # inter-agent discussion: tell EM what is valid AND ask PM, via the same
        # mid-phase gap channel, for concrete guidance on the next action.
        if tool_name != "read_store_variable" and tool_name not in phase.tools:
            available = list(phase.tools.keys())
            em_board.append({
                "invalid_tool": (
                    f"'{tool_name}' is not a callable tool in this phase. You MUST "
                    f"choose exactly one tool from {available} (or read_store_variable). "
                    f"Do not invent tool names or return a non-tool placeholder. If you "
                    f"believe the phase is finished, call its terminal tool; if you are "
                    f"blocked, explain why via a valid tool."
                )
            })
            if enable_gap_trigger and gap_interventions < gap_max_interventions:
                try:
                    gap_state = compute_deployment_state(
                        phase_manager.system_prompt.get("available_phases", []),
                        variable_store.get_all(),
                        snapshot_invalidated(),
                    )
                    if logger:
                        logger.start_step("phase_manager", f"pm_gap_invalidtool_{phase_name}_{execution_turns}")
                    gap = phase_manager.gap_check(
                        phase_description=phase.description,
                        goal=phase.goal,
                        board=em_board,
                        deployment_state=gap_state,
                    )
                    if logger:
                        logger.end_step(f"pm_gap_invalidtool_{phase_name}_{execution_turns}")
                    print(phase_name, "pm_gap(invalid_tool)", gap)
                    if gap.get("intervene") and gap.get("instruction"):
                        em_board.append({"pm_guidance": gap["instruction"]})
                except Exception as e:
                    print(phase_name, "pm_gap(invalid_tool) failed:", e)
                finally:
                    gap_interventions += 1
                    last_gap_turn = execution_turns
            continue

        # --- Consistency-gated supervision -----------------------------------
        # PM intervenes on EM's PROPOSED action iff it is inconsistent with the
        # global artifact-state belief (redundant / premature / stale) or EM has
        # stalled. The DECISION is deterministic (consistency_gate over the
        # belief); the LLM gap_check is invoked ONLY to write the instruction
        # CONTENT once the gate has fired. Skipped for read_store_variable (a
        # meta-op) and on a boundary/terminal tool (phase is ending anyway).
        if (enable_gap_trigger and tool_name != "read_store_variable"
                and not boundary_tool_used
                and gap_interventions < gap_max_interventions
                and (execution_turns - last_gap_turn) > gap_cooldown):
            available_phases_ = phase_manager.system_prompt.get("available_phases", [])
            all_vars_ = variable_store.get_all()
            invalidated_ = snapshot_invalidated()
            dep_state_ = compute_deployment_state(available_phases_, all_vars_, invalidated_)
            cur_status_ = next(
                (e["status"] for e in dep_state_
                 if correct_phase_name(e["phase"]) == phase_name),
                "not_started",
            )
            try:
                tool_inputs_ = list(phase.tool_arguments(tool_name))
            except Exception:
                tool_inputs_ = []
            stall_ = gap_signal_fired(em_board, gap_repeat_threshold, gap_error_threshold)
            intervene_, fired_ = consistency_gate(
                tool_inputs=tool_inputs_,
                current_phase_status=cur_status_,
                all_vars=all_vars_,
                invalidated=invalidated_,
                known=known_artifacts(available_phases_),
                stall_fired=stall_,
                policy=intervention_policy,
                random_gate_p=random_gate_p,
                rng=_gate_rng,
            )
            # --- offline-ablation telemetry ----------------------------------
            # Emit ONE machine-readable record per gate EVALUATION (both fire and
            # no-fire), capturing the proposed action + full belief snapshot. This
            # is what lets the never/random/always/uncertainty policies be
            # reconstructed EXACTLY offline from a single consistency run, and it
            # marks the trajectory-divergence turns for pruned re-runs. Pure log;
            # does not affect the decision above.
            if logger:
                known_ = known_artifacts(available_phases_)
                logger.log_event(
                    "gate_decision",
                    em_turn=execution_turns,
                    proposed_tool=tool_name,
                    tool_inputs=[a for a in tool_inputs_ if a in known_],
                    belief={a: artifact_status(all_vars_, invalidated_, a) for a in known_},
                    current_phase_status=cur_status_,
                    stall_fired=bool(stall_),
                    policy=intervention_policy,
                    fired=fired_,
                    intervene=bool(intervene_),
                )
            if intervene_:
                for p_ in fired_:
                    gate_stats[p_] = gate_stats.get(p_, 0) + 1
                if logger:
                    logger.start_step("phase_manager", f"pm_gap_{phase_name}_{execution_turns}")
                gap = phase_manager.gap_check(
                    phase_description=phase.description,
                    goal=phase.goal,
                    board=em_board + [{"gate_fired": fired_}],
                    deployment_state=dep_state_,
                )
                if logger:
                    logger.end_step(f"pm_gap_{phase_name}_{execution_turns}")
                print(phase_name, "consistency_gate fired", fired_, "pm_gap", gap)
                if gap.get("intervene"):
                    em_board.append({"pm_guidance": gap.get("instruction", ""), "gate_fired": fired_})
                    gate_stats["interventions"] += 1
                    gap_interventions += 1
                    last_gap_turn = execution_turns

        if tool_name == "read_store_variable": # give the variable to em
            try:
                name_list = [item["value"] for item in em_arguments]
                found, variable_value = get_arguments(variable_store, tmp_variable_store, {}, name_list)
                if not found:
                    variable_value_missing = variable_value.get('missing')
                    em_board.append({"error": f"Variable {variable_value_missing} not found in store"})
                    continue
                else:
                    for variable_name, variable_value in variable_value.items():
                        em_called_arguments.append({"key": variable_name, "value": variable_value})
                    continue
            except Exception as e:
                em_board.append({"error": f"Error when searching for variables in store: {e}"})
                continue
        else: # use the tool
            tool_arguments = phase.tool_arguments(tool_name)
            found, variable_value = get_arguments(variable_store, tmp_variable_store, em_arguments, tool_arguments)
            if not found:
                variable_value_missing = variable_value.get('missing')
                em_board.append({"error": f"Variable {variable_value_missing} not found in store when running {tool_name}"})
            else:
                try:
                    print("tool using: ", tool_name)
                    if logger:
                        logger.start_step("tool", tool_name)
                    observation = phase.tools[tool_name](**variable_value)
                    if logger:
                        logger.end_step(tool_name)
                    print("observation", observation)
                except Exception as e:
                    em_board.append(_truncate_entry({"error": f"Error when running {tool_name}: {e}"}))
                    # Non-monotonic Update: attribute a root cause and demote the
                    # responsible EARLIER phase's artifacts present -> invalidated.
                    if enable_retroactive_invalidation:
                        rc_ = attribute_root_cause(str(e))
                        if rc_ and _phase_precedes(rc_, phase_name):
                            mark_phases_invalid([rc_])
                            gate_stats["retroactive_invalidations"] += 1
                            print(phase_name, "retroactive_invalidation ->", rc_)
                    continue
                for item in observation: # store the observation
                    storage = item["storage"]
                    variable_name = item["variable_name"]
                    value = item["value"]

                    if storage == "temporary":
                        em_board.append(f"successfully run {tool_name}.")
                        tmp_variable_store.write(variable_name, value)
                        if judge_store_variable(value):
                            arguments_board.append(_truncate_entry({variable_name: value}))
                            arguments_board.append("\n")
                        else:
                            em_board.append(f"Variable {variable_name} is successfully stored.\n")
                            em_board.append(f"If you HAVE TO get the value of {variable_name}, you can set the tool_name as `read_store_variable` and the variable_name as {variable_name} in the arguments of the next tool. \n")
                            em_board.append(f"You should try your best to AVOID using the `read_store_variable` tool.")
                            arguments_board.append("\n")

                    elif storage == "permanent":
                        em_board.append(f"successfully run {tool_name}.")
                        variable_store.write(variable_name, value)
                        phase_observation[variable_name] = value
                        if judge_store_variable(value):
                            arguments_board.append(_truncate_entry({variable_name: value}))
                            arguments_board.append("\n")
                        else:
                            em_board.append(f"Variable {variable_name} is successfully stored. \n")
                            em_board.append(f"If you HAVE TO get the value of {variable_name}, you can set the tool_name as `read_store_variable` and the variable_name as {variable_name} in the arguments of the next tool. \n")
                            em_board.append(f"You should try your best to AVOID using the `read_store_variable` tool.")
                            arguments_board.append("\n")

                    elif storage == "error":
                        em_board.append(_truncate_entry(f"error when running {tool_name}: {value}"))
                        arguments_board.append("\n")
                        if enable_retroactive_invalidation:
                            rc_ = attribute_root_cause(str(value))
                            if rc_ and _phase_precedes(rc_, phase_name):
                                mark_phases_invalid([rc_])
                                gate_stats["retroactive_invalidations"] += 1
                                print(phase_name, "retroactive_invalidation ->", rc_)

                em_board.append(f"Choose another step.")

                if boundary_tool_used or execution_turns == max_execution_turns:
                    tmp_pm_board = pm_board.copy()
                    phase_sum = phase_summarizer.summarize(em_board, observation=phase_observation)
                    tmp_pm_board.append({"phase_summary": phase_sum})
                    # Persistent belief state: which phases' required_outputs are
                    # already satisfied. Keeps PM from looping back to a phase
                    # that is provably complete.
                    tmp_pm_board.append({"deployment_state": compute_deployment_state(
                        phase_manager.system_prompt.get("available_phases", []),
                        variable_store.get_all(),
                        snapshot_invalidated(),
                    )})

                    # PM decision, wrapped in a cross-agent negotiation loop: on a
                    # need_retry, PM's reason is an INSTRUCTION to EM. EM may accept
                    # it, negotiate (push back with first-hand observations so PM
                    # revises), or reject it (proceed on its own plan). negotiate
                    # re-runs PM with EM's feedback, bounded by max_negotiation_rounds.
                    negotiation_rounds = 0
                    pm_loop_action = "break"  # break | retry | fatal
                    while True:
                        if logger:
                            logger.start_step("phase_manager", f"pm_decision_{phase_name}")
                        pm_response = phase_manager.run(board=tmp_pm_board)
                        if logger:
                            logger.end_step(f"pm_decision_{phase_name}")
                        pm_decision = pm_response
                        pm_board.append(pm_decision)
                        decision = pm_decision["decision"]
                        reason = pm_decision["reason"]
                        next_phase = pm_decision["next_phase"]
                        final_decision = decision
                        last_pm_decision = pm_decision
                        # Persist any "this earlier phase is defective" flags so
                        # the deployment_state shows them as needs_redo until the
                        # phase is actually re-run (prevents flip-flop where the
                        # stale outputs look completed again next turn).
                        mark_phases_invalid(pm_decision.get("invalidate_phases"))

                        print("pm_decision", pm_decision)
                        print("decision", decision)
                        print("next_phase", next_phase)

                        if decision == "need_retry":
                            if enable_negotiation and enable_override and negotiation_rounds < max_negotiation_rounds:
                                if logger:
                                    logger.start_step("execution_master", f"em_instruction_{phase_name}_{negotiation_rounds}")
                                em_instr = execution_master.respond_to_instruction(
                                    instruction=reason,
                                    phase_description=phase.description,
                                    goal=phase.goal,
                                    board=em_board,
                                    tool_list=phase.tools_schemas,
                                )
                                if logger:
                                    logger.end_step(f"em_instruction_{phase_name}_{negotiation_rounds}")
                                em_inst_decision = str(em_instr.get("decision", "accept")).strip().lower()
                                em_inst_message = em_instr.get("message", "")
                                print("em_instruction", em_instr)

                                if em_inst_decision == "negotiate":
                                    negotiation_rounds += 1
                                    pm_board.append({"em_negotiation": em_inst_message})
                                    tmp_pm_board.append({"pm_instruction": reason})
                                    tmp_pm_board.append({
                                        "em_negotiation": em_inst_message,
                                        "note": "EM pushed back on the instruction above using first-hand tool observations. Revise your decision/instruction accordingly, or confirm need_retry if EM's objection is unfounded.",
                                    })
                                    continue  # re-run PM with EM's feedback
                                if em_inst_decision == "reject":
                                    em_board.append({"info": "pm_decision=need_retry; EM rejected the instruction and continues its own plan"})
                                    em_board.append({"em_rejection": em_inst_message})
                                    pm_board.append({"em_rejection": em_inst_message})
                                    pm_loop_action = "retry"
                                    break
                                # accept (or any unrecognized value -> treat as accept)
                                em_board.append({"info": "pm_decision=need_retry; EM accepted the instruction"})
                                em_board.append({"reason for need_retry": reason})
                                if em_inst_message:
                                    em_board.append({"em_acceptance": em_inst_message})
                                pm_loop_action = "retry"
                                break
                            else:
                                # negotiation disabled or budget exhausted -> original behavior
                                em_board.append({"info": "pm_decision=need_retry, retry current loop"})
                                em_board.append({"reason for need_retry": reason})
                                pm_loop_action = "retry"
                                break

                        if decision in ("ok", "blocked"):
                            pm_loop_action = "break"
                            break
                        if decision == "fatal":
                            pm_loop_action = "fatal"
                            break
                        # unrecognized decision -> exit the phase safely
                        pm_loop_action = "break"
                        break

                    if pm_loop_action == "fatal":
                        raise Exception("Deployment fatal, please try again")
                    if pm_loop_action == "break":
                        break  # exit the EM while loop
                    continue  # pm_loop_action == "retry": iterate the EM while loop
    
    print("em_board", em_board)
    print(phase_name, "gate_stats", gate_stats)

    if logger:
        # Persist per-phase mechanism statistics to calls.jsonl (previously only
        # printed). Enables per-predicate supervision-cost analysis directly from
        # the log, with no re-run. Stamped while the phase context is still live.
        logger.log_event("gate_stats", **gate_stats)
        logger.end_phase(phase_name)

    
    phase_pm_board_conclusion = phase_summarizer.summarize(em_board, observation=phase_observation)
    print("phase_pm_board_conclusion", phase_pm_board_conclusion)
    print("next_phase", next_phase)
    print("final_decision", final_decision)
    print("last_pm_decision", last_pm_decision)
    # 4th element is a status dict so we can add fields later without
    # breaking tuple positions for existing consumers (item["result"][1/2]).
    return True, next_phase, phase_pm_board_conclusion, {"final_decision": final_decision, "pm_decision": last_pm_decision, "gate_stats": gate_stats}


def main():
    parser = argparse.ArgumentParser(description='Run script with repository full name as an argument.')

    # Pipeline entry: either a paper PDF (RepoIngest will run the full
    # 5-tool flow read_pdf -> github_link_extract -> git_clone ->
    # task_selection) or a github link directly (RepoIngest skips the
    # PDF parsing half and jumps straight to git_clone). Exactly one is
    # required.
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pdf_path",
        type=str,
        help="path to the source paper PDF file (full ingest)",
    )
    input_group.add_argument(
        "--github_link",
        type=str,
        help="github.com repo URL to clone directly, skipping PDF parsing",
    )
    input_group.add_argument(
        "--resume_from",
        type=str,
        help=(
            "path to a variable_store_after_parallel.json checkpoint. "
            "Skips repoingest/weightresolve/dockersetup and starts at the "
            "post-parallel phase (typically apiadaptation) with the variable "
            "store + pm_board restored from the checkpoint."
        ),
    )

    parser.add_argument("--test_file_dir", type=str, help="path to the test file directory", required=True)
    parser.add_argument("--backend", type=str, help="backend to use", required=True)
    args = parser.parse_args()

    # os.path.abspath is a no-op for already-absolute paths and resolves
    # relative paths against cwd, so we can call it unconditionally.
    test_file_dir = Path(os.path.abspath(args.test_file_dir))
    backend = args.backend
    if backend not in ["openai", "gr"]:
        raise ValueError("Invalid backend. Please choose a valid backend.")

    # Initialize pipeline logger; name the run with the input source so
    # logs are easy to identify.
    # Name the run after the test subject: test_file_dir is typically
    # .../test_files/<model_name>/case_*, so parent.name gives the model name.
    source_name = test_file_dir.parent.name
    run_id = f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pipeline_logger = init_logger(run_id=run_id)
    print(f"[main] Logging to {pipeline_logger.log_dir}")

    pm_board = []
    phase_manager = PhaseManager(backend=backend)
    phase_name_list = ["repoingest"]
    phase_change_turns = 0

    if args.resume_from:
        ckpt_path = os.path.abspath(args.resume_from)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--resume_from checkpoint not found: {ckpt_path}")
        ckpt = _load_parallel_checkpoint(ckpt_path)
        for k, v in (ckpt.get("variable_store") or {}).items():
            variable_store.write(k, v)
        pm_board = list(ckpt.get("pm_board") or [])
        phase_name_list = [correct_phase_name(p) for p in (ckpt.get("phase_name_list") or ["apiadaptation"])]
        phase_change_turns = int(ckpt.get("phase_change_turns") or 0)
        # test_file_dir passed on this invocation may differ from the one
        # captured at save time. Prefer the live CLI value so the resume run
        # can point at a different fixture.
        variable_store.write("test_file_dir", test_file_dir)
        print(
            f"[main] resumed from checkpoint: "
            f"variables={list((ckpt.get('variable_store') or {}).keys())}, "
            f"next_phases={phase_name_list}, "
            f"phase_change_turns={phase_change_turns}"
        )
    elif args.pdf_path:
        pdf_path = os.path.abspath(args.pdf_path)
        variable_store.write("pdf_path", pdf_path)
        pm_board.append({"pdf_path": pdf_path})
    else:
        github_link = args.github_link.strip()
        # Cheap CLI sanity check - the strict regex validation lives inside
        # RepoIngest._extract_first_github_link, this just rejects obvious
        # typos at the entry point so we don't waste a phase turn on them.
        if "github.com" not in github_link:
            raise ValueError(
                "--github_link must contain 'github.com' "
                "(e.g. https://github.com/owner/repo)"
            )
        variable_store.write("github_link", github_link)
        pm_board.append({"github_link": github_link})
        # Hint to phase_manager that the PDF half is unnecessary on this run.
        # RepoIngest's EM will additionally see github_link in arguments_board
        # via variable_store.get_all() and naturally pick git_clone first
        # without read_pdf / github_link_extract_*.
        
        # pm_board.append(
        #     "github_link was provided directly via --github_link. "
        #     "RepoIngest can skip read_pdf / github_link_extract_with_text / "
        #     "github_link_extract_with_llm and go straight to git_clone, "
        #     "then task_selection."
        # )

    # pdf_path = "2403.07636v4.pdf"
    # backend = "gr"
    # test_file_dir = Path("/home/user/test_files")

    # repo_root = Path("/home/user/repo")
    # variable_store.write("repo_root", repo_root)
    # pm_board.append({"repo_root": repo_root})
    # phase_name_list = ["weightresolve"]
    # phase_name_list = ["weightresolve", "DockerSetUp"]


    while phase_change_turns < max_phase_turns: # run pm
        phase_change_turns += 1
        parallel = False
        if len(phase_name_list) > 1:
            parallel = True

        variable_store.write("test_file_dir", test_file_dir)
        pm_board.append({"test_file_dir": test_file_dir})



        # Snapshot the phase set we're about to run - after the parallel
        # block, phase_name_list is overwritten with the PM's next-phase
        # decision, so we need this to detect "we just finished the
        # weightresolve + dockersetup parallel" for the checkpoint save.
        pre_parallel_phase_set = set(phase_name_list)

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(execute_phase, phase_name_item, phase_manager, backend, pm_board.copy(), phase_name_list): phase_name_item for phase_name_item in phase_name_list}
            done, not_done = wait(futures, return_when=ALL_COMPLETED)

            for future in done:
                phase_name_item = futures[future]
                try:
                    result = future.result()   # the exception is actually raised here
                except Exception as e:
                    print(f"\n❌ Error in phase: {phase_name_item}")
                    print(f"Exception: {e}")
                    import traceback
                    traceback.print_exc()

            results = []
            for fut, pn in futures.items():
                try:
                    r = fut.result()
                except Exception as e:
                    r = (
                        False,
                        [],
                        {
                            "phase_name": pn,
                            "status": "error",
                            "summary_for_pm": str(e),
                        },
                        # An unhandled exception out of execute_phase is
                        # treated as fatal so the outer loop won't silently
                        # retry it forever.
                        {"final_decision": "fatal"},
                    )
                results.append({"phase": pn, "result": r})
        if not parallel:
            for item in results:
                pm_board.append(item["result"][2])
                status = item["result"][3] if len(item["result"]) > 3 else {}
                if status.get("pm_decision"):
                    pm_board.append(status["pm_decision"])
                phase_name_list = item["result"][1]
                phase_name_list = [correct_phase_name(p) for p in phase_name_list]
                # servicedelivery is the terminal phase of the pipeline (it
                # absorbed the old runtimevalidation + endphase pair). But
                # "execute_phase returned" is not the same as "deployment
                # finished": the phase may have hit max_execution_turns
                # mid-validation, PM may have asked for a retry, or PM may
                # have decided to loop back to a fix-up phase (e.g. CUDA
                # mismatch -> redo dockersetup). Only stop the pipeline on
                # a clean PM-confirmed finish.
                if item["phase"] == "servicedelivery":
                    status = item["result"][3] if len(item["result"]) > 3 else {}
                    final_decision = status.get("final_decision")

                    if final_decision == "ok" and not phase_name_list:
                        print("[main] servicedelivery completed; pipeline done.")
                        if pipeline_logger:
                            pipeline_logger.write_final_summary()
                        return

                    if final_decision in ("blocked", "fatal"):
                        print(
                            f"[main] servicedelivery {final_decision} by PM; "
                            "stopping pipeline."
                        )
                        if pipeline_logger:
                            pipeline_logger.write_final_summary()
                        return

                    # final_decision in {None, 'need_retry', 'ok' with non-empty
                    # next_phase}: keep iterating. If PM did not give us any
                    # next phase to run, default to retrying servicedelivery so
                    # the outer loop has something to run instead of spinning
                    # on an empty phase_name_list. The max_phase_turns guard
                    # on the outer while still bounds total retries.
                    if not phase_name_list:
                        phase_name_list = ["servicedelivery"]
                        print(
                            f"[main] servicedelivery did not finalize "
                            f"(final_decision={final_decision!r}); scheduling retry."
                        )
                    # else: PM gave us next phases, fall through and let the
                    # outer loop pick them up.

                if any(p in allowed_parallel_phases for p in phase_name_list):
                    parallel_discriminator = ParallelDiscriminator(backend=backend)
                    phase_name_list = parallel_discriminator.run(pm_board)
        else:
            next_phase_list = []
            for item in results:
                next_phase_list.append(f"{item["result"][2]}: suggested_next_phase {item["result"][1]}")
                status = item["result"][3] if len(item["result"]) > 3 else {}
                if status.get("pm_decision"):
                    pm_board.append(status["pm_decision"])
            pm_board.append("You have run the parallel execution phases, suggested next phases are:")
            pm_board.append(next_phase_list)
            pm_board.append({"deployment_state": compute_deployment_state(
                phase_manager.system_prompt.get("available_phases", []),
                variable_store.get_all(),
                snapshot_invalidated(),
            )})
            pm_board.append("Now you have to decide the next phase to run.")
            pm_response = phase_manager.run(board=pm_board)
            pm_decision = pm_response
            mark_phases_invalid(pm_decision.get("invalidate_phases"))
            phase_name_list = pm_decision["next_phase"]
            phase_name_list = [correct_phase_name(p) for p in phase_name_list]
            if any(p in allowed_parallel_phases for p in phase_name_list):
                parallel_discriminator = ParallelDiscriminator(backend=backend)
                phase_name_list = parallel_discriminator.run(pm_board)
            pm_board.append(pm_decision)
            print("phase_name_list", phase_name_list)
            print("pm_board", pm_board)

            if pre_parallel_phase_set == set(allowed_parallel_phases):
                _save_parallel_checkpoint(
                    Path(pipeline_logger.log_dir),
                    variable_store,
                    pm_board,
                    phase_name_list,
                    phase_change_turns,
                )


if __name__ == "__main__":
    main()
