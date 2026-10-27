"""Agent 6 - Decision Agent.

Assembles the four-way verdict from the uncertainty vector. It never sees a
single collapsed score, which is deliberate: it must justify *which* dimension
drove the call.
"""

from src.core import uncertainty as unc
from src.core import verdict as verdict_policy
from src.core.schemas import VerdictReport
from src.core.state import log


def run(state):
    evidence = state.get("evidence", [])
    computations = state.get("computations", [])

    vector = state.get("uncertainty") or unc.build(evidence, computations)
    state["uncertainty"] = vector

    outcome, rationale, reasons = verdict_policy.decide(evidence, computations, vector)
    overall = unc.overall(vector)

    report = VerdictReport(
        verdict=outcome,
        overall_uncertainty=overall,
        uncertainty=vector.as_dict(),
        dominant_dimension=vector.dominant(),
        rationale=rationale,
        support_score=verdict_policy.support_fraction(evidence),
        evidence=[e.to_dict() for e in evidence],
        computations=[c.to_dict() for c in computations],
        abstain_reasons=reasons,
        trace=state.get("trace", []),
    )
    state["report"] = report
    return log(state, "decision_agent", f"{outcome} (uncertainty {overall:.2f})")
