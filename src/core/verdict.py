"""Turns the uncertainty vector + evidence balance into one of four verdicts.

The four-way output (Supported / Contradicted / Uncertain / Abstain) is the
whole point: a binary yes/no would hide the difference between "we looked and
found against it" and "we could not look properly".
"""

import config
from src.core import uncertainty as unc


def support_fraction(evidence) -> float:
    """Weighted share of supporting evidence, in [0, 1]."""
    total = sum(e.weight for e in evidence)
    if total <= 0:
        return 0.0
    pos = sum(e.weight for e in evidence if e.polarity > 0)
    return round(pos / total, 4)


def abstain_reasons(vector, evidence) -> list:
    """Explicit, checkable criteria - not a hidden threshold."""
    reasons = []
    if vector.evidence_coverage >= 0.7:
        reasons.append(
            "evidence coverage is too thin: "
            f"uncertainty {vector.evidence_coverage:.2f} >= 0.70"
        )
    if vector.graph_support >= 0.9 and not any(
        e.source == "knowledge_graph" for e in evidence
    ):
        reasons.append("knowledge graph returned no path for this drug-target pair")
    if not any(e.source == "assay" for e in evidence):
        reasons.append("no assay-level evidence available to test the claim")
    return reasons


def decide(evidence, computations, vector) -> tuple:
    """Returns (verdict, rationale, abstain_reasons)."""
    overall = unc.overall(vector)
    reasons = abstain_reasons(vector, evidence)
    support = support_fraction(evidence)
    contra_share = vector.literature_conflict

    if reasons and overall >= config.ABSTAIN_THRESHOLD:
        rationale = (
            "Abstaining: the evidence base cannot resolve this claim, so the "
            f"framework refuses to rank it. Dominant doubt: {vector.dominant()}."
        )
        return "ABSTAIN", rationale, reasons

    if contra_share >= 0.6 and support < 0.5:
        rationale = (
            f"Contradicted: {contra_share:.0%} of retrieved literature conflicts "
            f"with the claim and only {support:.0%} of weighted evidence supports it."
        )
        return "CONTRADICTED", rationale, reasons

    if contra_share >= 0.35:
        rationale = (
            f"Uncertain: the literature is genuinely split "
            f"({contra_share:.0%} contradicting, {support:.0%} of weighted evidence "
            "in favour), so the claim cannot be called either way yet."
        )
        return "UNCERTAIN", rationale, reasons

    if overall >= config.ABSTAIN_THRESHOLD:
        rationale = (
            f"Uncertain: overall uncertainty {overall:.2f} is above the "
            f"{config.ABSTAIN_THRESHOLD:.2f} decision threshold, driven by "
            f"{vector.dominant()}."
        )
        return "UNCERTAIN", rationale, reasons

    if support >= config.SUPPORT_THRESHOLD:
        rationale = (
            f"Supported: {support:.0%} of weighted evidence agrees, models are "
            f"consistent and overall uncertainty is {overall:.2f}."
        )
        return "SUPPORTED", rationale, reasons

    rationale = (
        f"Uncertain: evidence exists but is not lopsided enough to support the "
        f"claim ({support:.0%} in favour, overall uncertainty {overall:.2f})."
    )
    return "UNCERTAIN", rationale, reasons
