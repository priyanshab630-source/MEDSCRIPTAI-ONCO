"""Both orchestration engines produce the same verdict."""

import pytest

import config
from src.agents.workflow import run_pipeline
from src.core import engine
from src.core.schemas import Hypothesis

HYPOTHESIS = Hypothesis(drug="SHP099", target="SHP2",
                        disease="KRAS-mutant NSCLC",
                        smiles="Nc1nc(N2CCNCC2)c2cc(Cl)ccc2n1")


def _run(use_langgraph: bool, monkeypatch):
    monkeypatch.setattr(config, "USE_LANGGRAPH", use_langgraph)
    return run_pipeline(HYPOTHESIS, trace=False)


def test_builtin_engine_reports_itself(monkeypatch):
    report = _run(False, monkeypatch)
    assert report.engine == "builtin"


def test_langgraph_engine_matches_builtin(monkeypatch):
    pytest.importorskip("langgraph", reason="langgraph is an optional extra")

    builtin = _run(False, monkeypatch)
    langgraph_report = _run(True, monkeypatch)

    assert langgraph_report.engine == "langgraph"
    assert langgraph_report.verdict == builtin.verdict
    assert langgraph_report.overall_uncertainty == builtin.overall_uncertainty
    assert len(langgraph_report.evidence) == len(builtin.evidence)


def test_langgraph_handles_the_adaptive_loop(monkeypatch):
    pytest.importorskip("langgraph", reason="langgraph is an optional extra")

    monkeypatch.setattr(config, "USE_LANGGRAPH", True)
    report = run_pipeline(Hypothesis(drug="CompoundX", target="SHP2"), trace=False)
    assert report.verdict == "ABSTAIN"


def test_engine_name_is_reported():
    assert engine.engine_name()
