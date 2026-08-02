"""Coordination mechanisms for MADE (target design).

Centralizes the three named coordination mechanisms so that (a) the paper's
formalism matches the code exactly, and (b) ablations are a single config
switch instead of forked code paths:

  * ArtifactBelief / artifact_status -- the deterministic three-valued global
      artifact-state belief (Zhat): each artifact is ABSENT | PRESENT |
      INVALIDATED, computed with ZERO LLM calls from the shared variable store
      plus an invalidation set.
  * consistency_gate -- the four checkable predicates that decide WHEN the
      Phase Manager intervenes on the Execution Master's PROPOSED action:
        P_redundant  (re-producing already-valid work)
        P_premature  (a required input artifact is still absent)
        P_stale      (a required input artifact is invalidated / known-broken)
        P_stall      (progress has stalled: tool-repeat or consecutive errors)
      Intervene iff any predicate fires. The LLM is invoked only to WRITE the
      instruction content once this deterministic gate has fired.
  * attribute_root_cause -- maps a failure signature to the phase most likely
      responsible; this drives the non-monotonic Update operator (retroactive
      invalidation) in main.execute_phase.

All functions are pure and dependency-light: they operate on plain dicts / sets
passed in from main.execute_phase, so this module imports nothing from the
pipeline and cannot create import cycles. `policy` selects the ablation variant.
"""

import random as _random

# --- three-valued artifact status -----------------------------------------
ABSENT = "absent"
PRESENT = "present"
INVALIDATED = "invalidated"

# --- intervention policies (ablation switch) -------------------------------
#   consistency : the four predicates (full MADE)
#   never       : never intervene mid-phase (single-agent-per-phase lower bound)
#   boundary    : same gate as `never`; the phase-BOUNDARY PM decision (always
#                 on) reproduces the old "review only at the boundary" design
#   always      : intervene every EM turn (over-supervision upper bound)
#   random      : intervene with probability random_gate_p (frequency-matched
#                 control that isolates the VALUE of the consistency signal)
#   uncertainty : intervene when the EM action's uncertainty exceeds a threshold
#                 (requires logprob plumbing; inert until action_uncertainty is
#                 passed in -- see main.py)
POLICIES = ("consistency", "never", "boundary", "always", "random", "uncertainty")


def artifact_status(all_vars, invalidated, name):
    """Three-valued status of a single artifact (belief Zhat over one key)."""
    if name in invalidated:
        return INVALIDATED
    val = all_vars.get(name, None)
    if val not in (None, "", [], {}):
        return PRESENT
    return ABSENT


def known_artifacts(available_phases, extra=()):
    """Set of cross-phase artifact names.

    Restricting the premature/stale predicates to this set is what stops a
    literal tool argument (e.g. 'hint', 'path') from being mis-flagged as a
    missing cross-phase artifact.
    """
    s = set(extra)
    for ph in available_phases or []:
        for a in (ph.get("required_outputs") or []):
            s.add(a)
    return s


# --- root-cause attribution (drives the non-monotonic Update operator) ------
# Conservative, auditable keyword rules mapping a failure signature to the
# phase whose output is the likely root cause. First match wins. Advisory: the
# caller only ever invalidates an EARLIER phase and the PM confirms re-routing.
_ROOT_CAUSE_RULES = (
    (("cuda", "cudnn", "libcudnn", "no kernel image", "nvidia", "torch.cuda",
      "device-side assert", "gpu is not available"), "dockersetup"),
    (("safetensors", "state_dict", "checkpoint", "missing key", "size mismatch",
      ".bin", ".pt", ".pth", ".ckpt", "no such file or directory"), "weightresolve"),
    (("schema mismatch", "contract", "unexpected field", "missing required field"),
     "apiadaptation"),
)


def attribute_root_cause(error_text):
    """Map a failure signature to the responsible phase name, or None."""
    if not error_text:
        return None
    t = str(error_text).lower()
    for keywords, phase in _ROOT_CAUSE_RULES:
        if any(k in t for k in keywords):
            return phase
    return None


def consistency_gate(*, tool_inputs, current_phase_status,
                     all_vars, invalidated, known, stall_fired,
                     policy="consistency", random_gate_p=0.15, rng=None,
                     action_uncertainty=None, uncertainty_thr=0.5):
    """Decide whether PM should intervene on EM's proposed action.

    Returns (intervene: bool, fired: list[str]) where `fired` names the
    predicate(s) that triggered (used to condition the instruction and for
    mechanism statistics).

    Parameters
    ----------
    tool_inputs : iterable[str]
        Input-argument names of the proposed tool (phase.tool_arguments(tool)).
    current_phase_status : str
        'completed' | 'partial' | 'not_started' | 'needs_redo' for the phase
        currently executing (from compute_deployment_state).
    all_vars, invalidated : dict, set
        Shared variable store snapshot and the invalidation set (=> the belief).
    known : set[str]
        Cross-phase artifact names; restricts premature/stale.
    stall_fired : bool
        Precomputed P_stall (main reuses its existing gap_signal_fired).
    """
    policy = (policy or "consistency").lower()

    if policy in ("never", "boundary"):
        return False, []
    if policy == "always":
        return True, ["always"]
    if policy == "random":
        r = (rng or _random).random()
        return (r < random_gate_p), (["random"] if r < random_gate_p else [])
    if policy == "uncertainty":
        if action_uncertainty is None:
            return False, []  # requires logprob plumbing; inert until wired
        hit = action_uncertainty >= uncertainty_thr
        return hit, (["uncertainty"] if hit else [])

    # policy == "consistency": the four predicates over the belief Zhat
    fired = []
    inputs = [a for a in (tool_inputs or []) if a in known]

    # P_redundant: the current phase's required_outputs are all present & valid
    if current_phase_status == "completed":
        fired.append("redundant")
    # P_premature: a required cross-phase input artifact is still absent
    if any(artifact_status(all_vars, invalidated, a) == ABSENT for a in inputs):
        fired.append("premature")
    # P_stale: a required input artifact is invalidated (known-broken)
    if any(a in invalidated for a in inputs):
        fired.append("stale")
    # P_stall: progress has stalled (reuses main's existing heuristic)
    if stall_fired:
        fired.append("stall")

    return (len(fired) > 0), fired


def new_gate_stats():
    """Fresh per-phase mechanism-statistics counter."""
    return {
        "interventions": 0,
        "redundant": 0, "premature": 0, "stale": 0, "stall": 0,
        "always": 0, "random": 0, "uncertainty": 0,
        "retroactive_invalidations": 0,
    }
