"""End-to-end: the graph runs, terminates and produces a full report."""

from src.agents.workflow import build_graph, run_pipeline
from src.core.schemas import Hypothesis


def test_pipeline_supports_shp099():
    report = run_pipeline(Hypothesis(drug="SHP099", target="SHP2",
                                     disease="KRAS-mutant NSCLC",
                                     smiles="Nc1nc(N2CCNCC2)c2cc(Cl)ccc2n1"))
    assert report.verdict in ("SUPPORTED", "UNCERTAIN")
    assert 0.0 <= report.overall_uncertainty <= 1.0
    assert len(report.evidence) >= 3
    assert len(report.computations) == 3
    assert report.trace[-1]["agent"] == "decision_agent"


def test_pipeline_abstains_on_unknown_compound():
    report = run_pipeline(Hypothesis(drug="CompoundX", target="SHP2"))
    assert report.verdict == "ABSTAIN"
    assert report.abstain_reasons


def test_graph_terminates_without_smiles():
    report = run_pipeline(Hypothesis(drug="SHP099", target="SHP2"))
    assert report.computations == []


def test_every_agent_ran():
    state = build_graph().invoke({"hypothesis": Hypothesis("SHP099", "SHP2"),
                                  "loops": 0, "trace": []})
    agents = [step["agent"] for step in state["trace"]]
    for expected in ("hypothesis_planner", "evidence_agent", "computational_analysis",
                     "critic", "adaptive_planner", "decision_agent"):
        assert expected in agents
