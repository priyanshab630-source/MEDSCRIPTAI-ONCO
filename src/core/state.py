"""The shared TypedDict-style state that flows between the 6 agents."""

from typing import Any, TypedDict


class OncoState(TypedDict, total=False):
    hypothesis: Any            # schemas.Hypothesis
    plan: list                 # checks the Hypothesis Planner decided on
    evidence: list             # schemas.EvidenceItem
    computations: list         # schemas.ComputeResult
    uncertainty: Any           # schemas.UncertaintyVector
    contradictions: list       # human-readable falsification notes
    loops: int                 # adaptive re-planning rounds used
    next_action: str           # "evidence" | "decision"
    trace: list                # agent-by-agent audit trail
    report: Any                # schemas.VerdictReport


def log(state: OncoState, agent: str, message: str) -> OncoState:
    """Append one line to the audit trail (evidence traceability by design)."""
    state.setdefault("trace", []).append({"agent": agent, "message": message})
    return state
