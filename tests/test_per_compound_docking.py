"""Every analysed compound is docked when its target has a structure."""

import pytest

import config
from src.tools import docking_run, receptors


@pytest.fixture
def docking_on(monkeypatch):
    """Opt in to real docking for the tests that need it."""
    monkeypatch.setattr(config, "DOCKING_ENABLED", True)

pytest.importorskip("rdkit", reason="rdkit is optional")

LIGAND = "CCNC(=O)NCCC(=O)Nc1ccc2nc(C)sc2c1"


def test_registry_lists_the_demo_receptor():
    entries = {entry["target"]: entry for entry in receptors.registry()}
    assert "DEMO_4AJL" in entries
    assert entries["DEMO_4AJL"]["docked"] is True
    assert entries["DEMO_4AJL"]["box"]["center"]


def test_shp2_has_a_registered_structure():
    """SHP2 ships with the PDBbind 3O5X catalytic domain, so it docks for real."""
    entries = {entry["target"]: entry for entry in receptors.registry()}
    assert "SHP2" in entries
    assert entries["SHP2"]["docked"] is True


def test_targets_without_a_structure_report_unavailable():
    assert receptors.available("KRAS_G12C") is False
    assert docking_run.available("KRAS_G12C") is False


@pytest.mark.skipif(not (docking_run.available("DEMO_4AJL")
                        or __import__("src.tools.vina_adapter", fromlist=["x"]).available()),
                    reason="Vina/Meeko or the demo receptor is unavailable")
def test_compound_is_docked_and_returns_a_scene(docking_on):
    result = docking_run.dock_compound(LIGAND, "DEMO_4AJL", exhaustiveness=1)
    assert result["engine"].startswith("AutoDock Vina")
    assert result["best_affinity"] < 0
    assert result["poses"]
    assert result["own_analysis"]["counts"]["binding_site_residues"] > 0
    assert result["scene"]["counts"]["ligand_atoms"] > 10
    assert result["scene"]["counts"]["bonds"] > 10


def test_results_are_cached_and_deterministic(docking_on):
    if not docking_run.available("DEMO_4AJL"):
        pytest.skip("docking stack unavailable")
    first = docking_run.dock_compound(LIGAND, "DEMO_4AJL", exhaustiveness=1)
    second = docking_run.dock_compound(LIGAND, "DEMO_4AJL", exhaustiveness=1)
    assert first["best_affinity"] == second["best_affinity"]


def test_analysis_endpoint_docks_the_compound_it_was_given(docking_on):
    if not docking_run.available("DEMO_4AJL"):
        pytest.skip("docking stack unavailable")
    from fastapi.testclient import TestClient

    from api.app import app

    client = TestClient(app)
    body = client.post("/analyze", json={"drug": "probe", "target": "DEMO_4AJL",
                                         "disease": "", "smiles": LIGAND}).json()
    docking = body.get("docking") or {}
    assert docking.get("engine", "").startswith("AutoDock Vina")
    assert docking.get("best_affinity", 0) < 0
    assert docking["scene"]["counts"]["ligand_atoms"] > 10
