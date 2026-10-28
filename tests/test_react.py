"""ChemCrow-style tool orchestration (deterministic policy path)."""

from src.core.schemas import Hypothesis
from src.tools import react_agent

PROFILE = {"mw": [280, 500], "logp": [1.0, 5.8], "hbd": [0, 3], "hba": [4, 8],
           "tpsa": [50, 95], "pocket": {"hydrophobicity": 0.62, "polarity": 0.35,
                                        "reference_mw": 330}}


def _hypothesis():
    return Hypothesis(drug="SHP099", target="SHP2", disease="KRAS-mutant NSCLC",
                      smiles="Nc1nc(N2CCNCC2)c2cc(Cl)ccc2n1")


def test_loop_records_every_tool_call():
    result = react_agent.run(_hypothesis(), PROFILE, max_steps=4)
    assert result["mode"] in ("deterministic", "llm")
    tools = [step["tool"] for step in result["steps"]]
    assert "rdkit_descriptors" in tools
    assert any(t in tools for t in ("docking", "gnn_binding"))
    assert all("observation" in step for step in result["steps"])


def test_loop_stops_early_once_both_signals_exist():
    result = react_agent.run(_hypothesis(), PROFILE, max_steps=5)
    tools = [step["tool"] for step in result["steps"]]
    has_compute = any(t in tools for t in ("rdkit_descriptors", "gnn_binding", "docking"))
    has_evidence = any(t in tools for t in ("kg_lookup", "literature_search"))
    assert has_compute and has_evidence
    assert len(result["steps"]) <= 5


def test_respects_the_step_budget():
    result = react_agent.run(_hypothesis(), PROFILE, max_steps=2)
    assert len(result["steps"]) == 2


def test_no_smiles_skips_the_loop():
    result = react_agent.run(Hypothesis(drug="SHP099", target="SHP2"), PROFILE)
    assert result["mode"] == "skipped"
    assert result["steps"] == []


def test_unknown_tool_is_reported_not_raised():
    step = react_agent._run_tool("not_a_tool", _hypothesis(), PROFILE)
    assert step["ok"] is False
    assert "unknown tool" in step["observation"]
