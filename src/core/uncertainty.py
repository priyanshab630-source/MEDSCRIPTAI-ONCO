"""Builds the 6-dimension decomposed uncertainty vector.

The point of the framework is to *not* collapse everything into one confidence
number: the Decision agent needs to know whether the doubt comes from missing
evidence, from conflicting assays, from disagreeing models, or from the fact
that the models themselves are untrained prototypes.
"""

import config
from src.core.schemas import UncertaintyVector

# Evidence types a well-supported drug-target claim should have.
REQUIRED_EVIDENCE = ("literature", "knowledge_graph", "assay")

# Prototype models are not trained, so we carry a fixed reliability penalty.
# Set MODEL_RELIABILITY_PENALTY=0.15 once you plug in trained checkpoints.
MODEL_RELIABILITY_PENALTY = 0.45


def _clamp(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 4)


def evidence_coverage(evidence) -> float:
    """1.0 = nothing found, 0.0 = every required evidence type present."""
    if not evidence:
        return 1.0
    types = {e.source for e in evidence}
    found = len(types & set(REQUIRED_EVIDENCE))
    return _clamp(1.0 - found / len(REQUIRED_EVIDENCE))


def assay_consistency(evidence) -> float:
    """Disagreement between assay rows, scaled by how many rows exist."""
    assays = [e for e in evidence if e.source == "assay"]
    if not assays:
        return 0.9
    pos = sum(1 for a in assays if a.polarity > 0)
    neg = sum(1 for a in assays if a.polarity < 0)
    if pos + neg == 0:
        return 0.9
    minority = min(pos, neg) / (pos + neg)
    # Few rows = weak basis for consistency, so add a small sample penalty.
    sample_penalty = 0.2 if len(assays) < 2 else 0.0
    return _clamp(minority * 2 + sample_penalty)


def computational_agreement(computations) -> float:
    """Spread between the QSAR / GNN / docking scores."""
    if not computations:
        return 1.0
    scores = [c.score for c in computations]
    spread = max(scores) - min(scores)
    return _clamp(spread)


def graph_support(evidence) -> float:
    """1.0 = the knowledge graph says nothing about this pair."""
    kg = [e for e in evidence if e.source == "knowledge_graph"]
    if not kg:
        return 1.0
    support = sum(e.weight for e in kg if e.polarity > 0)
    against = sum(e.weight for e in kg if e.polarity < 0)
    if support + against == 0:
        return 1.0
    # Agreement ratio: +1 = every path supports, -1 = every path contradicts.
    agreement = (support - against) / (support + against)
    # A clean, agreeing subgraph drives uncertainty to ~0; a split one stays high.
    return _clamp(0.5 * (1.0 - agreement))


def literature_conflict(evidence) -> float:
    """Share of the retrieved literature that contradicts the hypothesis."""
    lit = [e for e in evidence if e.source == "literature"]
    if not lit:
        return 0.9
    contra = sum(1 for e in lit if e.polarity < 0)
    return _clamp(contra / len(lit))


def model_reliability() -> float:
    return MODEL_RELIABILITY_PENALTY


def build(evidence, computations) -> UncertaintyVector:
    return UncertaintyVector(
        evidence_coverage=evidence_coverage(evidence),
        assay_consistency=assay_consistency(evidence),
        computational_agreement=computational_agreement(computations),
        graph_support=graph_support(evidence),
        literature_conflict=literature_conflict(evidence),
        model_reliability=model_reliability(),
    )


def overall(vector: UncertaintyVector) -> float:
    return vector.overall(config.UNCERTAINTY_WEIGHTS)
