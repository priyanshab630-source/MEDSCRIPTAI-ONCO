"""Verdict policy: four outcomes, explicit abstain criteria."""

from src.core import verdict
from src.core.schemas import EvidenceItem, UncertaintyVector


def _vector(**overrides):
    base = dict(evidence_coverage=0.0, assay_consistency=0.0,
                computational_agreement=0.1, graph_support=0.2,
                literature_conflict=0.0, model_reliability=0.45)
    base.update(overrides)
    return UncertaintyVector(**base)


def test_support_fraction():
    evidence = [EvidenceItem("literature", "a", 1, 0.8),
                EvidenceItem("literature", "b", -1, 0.2)]
    assert verdict.support_fraction(evidence) == 0.8


def test_supported_when_evidence_agrees():
    evidence = [EvidenceItem("literature", "a", 1, 0.9),
                EvidenceItem("knowledge_graph", "b", 1, 0.6),
                EvidenceItem("assay", "c", 1, 0.7)]
    outcome, _, _ = verdict.decide(evidence, [], _vector())
    assert outcome == "SUPPORTED"


def test_contradicted_when_literature_opposes():
    evidence = [EvidenceItem("literature", "a", -1, 0.9),
                EvidenceItem("literature", "b", -1, 0.9)]
    outcome, _, _ = verdict.decide(evidence, [], _vector(literature_conflict=1.0))
    assert outcome == "CONTRADICTED"


def test_abstain_when_nothing_retrieved():
    vector = _vector(evidence_coverage=1.0, graph_support=1.0,
                     assay_consistency=0.9, literature_conflict=0.9,
                     computational_agreement=1.0)
    outcome, _, reasons = verdict.decide([], [], vector)
    assert outcome == "ABSTAIN"
    assert len(reasons) >= 2


def test_uncertain_when_mixed():
    evidence = [EvidenceItem("literature", "a", 1, 0.5),
                EvidenceItem("literature", "b", -1, 0.5)]
    outcome, _, _ = verdict.decide(
        evidence, [], _vector(literature_conflict=0.5, graph_support=0.5)
    )
    assert outcome in ("UNCERTAIN", "ABSTAIN")
