"""The six uncertainty dimensions and their bounds."""

from src.core import uncertainty as unc
from src.core.schemas import ComputeResult, EvidenceItem


def test_no_evidence_is_maximally_uncertain():
    vector = unc.build([], [])
    assert vector.evidence_coverage == 1.0
    assert vector.graph_support == 1.0
    assert unc.overall(vector) > 0.6


def test_full_evidence_lowers_uncertainty():
    evidence = [
        EvidenceItem("literature", "paper says yes", 1, 0.8),
        EvidenceItem("knowledge_graph", "A inhibits B", 1, 0.55),
        EvidenceItem("assay", "IC50 0.07 uM", 1, 0.7),
        EvidenceItem("assay", "pERK IC50 0.18 uM", 1, 0.7),
    ]
    computations = [
        ComputeResult("QSAR", 0.8, "fits"),
        ComputeResult("MolecularGNN", 0.82, "fits"),
        ComputeResult("Docking", 0.79, "fits"),
    ]
    vector = unc.build(evidence, computations)
    assert vector.evidence_coverage == 0.0
    assert vector.assay_consistency == 0.0
    assert vector.computational_agreement < 0.1
    assert unc.overall(vector) < 0.4


def test_disagreeing_models_raise_agreement_penalty():
    computations = [ComputeResult("QSAR", 0.9, ""), ComputeResult("Docking", 0.1, "")]
    assert unc.computational_agreement(computations) > 0.5


def test_split_literature_raises_conflict():
    evidence = [
        EvidenceItem("literature", "supports", 1, 0.5),
        EvidenceItem("literature", "contradicts", -1, 0.5),
    ]
    assert unc.literature_conflict(evidence) == 0.5


def test_overall_is_bounded():
    vector = unc.build(
        [EvidenceItem("literature", "x", 1, 1.0)], [ComputeResult("QSAR", 1.0, "")]
    )
    assert 0.0 <= unc.overall(vector) <= 1.0
