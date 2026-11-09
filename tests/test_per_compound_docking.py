"""Every analysed compound is docked when its target has a structure."""

import pytest

from src.tools import docking_run, receptors

pytest.importorskip("rdkit", reason="rdkit is optional")

LIGAND = "CCNC(=O)NCCC(=O)Nc1ccc2nc(C)sc2c1"


def test_registry_lists_the_demo_receptor():
    entries = {entry["target"]: entry for entry in receptors.registry()}
    assert "DEMO_4AJL" in entries
    assert entries["DEMO_4AJL"]["docked"] is True
    assert entries["DEMO_4AJL"]["box"]["center"]


def test_targets_without_a_structure_report_unavailable():
    assert receptors.available("SHP2") is False
    assert docking_run.available("SHP2") is False


@pytest.mark.skipif(not docking_run.available("DEMO_4AJL"),
                    reason="Vina/Meeko or the demo receptor is unavailable")
def test_compound_is_docked_and_returns_a_scene():
    result = docking_run.dock_compound(LIGAND, "DEMO_4AJL", exhaustiveness=1)
    assert result["engine"].startswith("AutoDock Vina")
    assert result["best_affinity"] < 0
    assert result["poses"]
    assert result["own_analysis"]["counts"]["binding_site_residues"] > 0
    assert result["scene"]["counts"]["ligand_atoms"] > 10
    assert result["scene"]["counts"]["bonds"] > 10


@pytest.mark.skipif(not docking_run.available("DEMO_4AJL"),
                    reason="Vina/Meeko or the demo receptor is unavailable")
def test_results_are_cached_and_deterministic():
    first = docking_run.dock_compound(LIGAND, "DEMO_4AJL", exhaustiveness=1)
    second = docking_run.dock_compound(LIGAND, "DEMO_4AJL", exhaustiveness=1)
    assert first["best_affinity"] == second["best_affinity"]


@pytest.mark.skipif(not docking_run.available("DEMO_4AJL"),
                    reason="Vina/Meeko or the demo receptor is unavailable")
def test_analysis_endpoint_docks_the_compound_it_was_given():
    from fastapi.testclient import TestClient

    from api.app import app

    client = TestClient(app)
    body = client.post("/analyze", json={"drug": "probe", "target": "DEMO_4AJL",
                                         "disease": "", "smiles": LIGAND}).json()
    docking = body.get("docking") or {}
    assert docking.get("engine", "").startswith("AutoDock Vina")
    assert docking.get("best_affinity", 0) < 0
    assert docking["scene"]["counts"]["ligand_atoms"] > 10
