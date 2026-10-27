"""Agent 5 - Adaptive Planner.

Decides whether the current evidence is enough to answer, or whether another
retrieval round is worth it. It is the only agent that can send the graph
backwards, and it stops after MAX_ADAPTIVE_LOOPS so a run always terminates.
"""

import config
from src.core import uncertainty as unc
from src.core.state import log


def run(state):
    evidence = state.get("evidence", [])
    computations = state.get("computations", [])
    loops = state.get("loops", 0)

    vector = unc.build(evidence, computations)
    overall = unc.overall(vector)
    state["uncertainty"] = vector

    coverage_gap = vector.evidence_coverage >= 0.4
    can_retry = loops < config.MAX_ADAPTIVE_LOOPS

    if coverage_gap and can_retry:
        state["loops"] = loops + 1
        state["next_action"] = "evidence"
        return log(
            state, "adaptive_planner",
            f"uncertainty {overall:.2f} with a coverage gap -> widening search "
            f"(loop {loops + 1}/{config.MAX_ADAPTIVE_LOOPS})",
        )

    state["next_action"] = "decision"
    return log(
        state, "adaptive_planner",
        f"stopping after {loops} retry round(s); overall uncertainty {overall:.2f}",
    )
