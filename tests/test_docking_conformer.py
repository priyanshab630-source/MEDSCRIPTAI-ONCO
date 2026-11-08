"""Conformer/MMFF docking layer and the Vina adapter."""

import sys
import types

import pytest

from src.tools import docking, vina_adapter

SHP2_POCKET = {
    "hydrophobicity": 0.62, "polarity": 0.35,
    "mw_range": [250, 500], "radius_range": [3.2, 6.6],
    "aromatic_rings_range": [1, 3], "reference_mw": 380,
}
SHP099 = "Nc1nc(N2CCNCC2)c2cc(Cl)ccc2n1"
ETHANOL = "CCO"


def test_engine_reports_the_conformer_backend():
    assert docking.engine() in ("conformer-mmff", "descriptor-surrogate")


@pytest.mark.skipif(docking.engine() != "conformer-mmff", reason="rdkit not installed")
def test_conformer_profile_is_physical():
    profile = docking.conformer_profile(SHP099)
    assert profile["n_conformers"] >= 1
    assert profile["min_energy"] > -1000
    assert profile["energy_spread"] >= 0
    assert profile["radius_of_gyration"] > 0
    assert profile["aromatic_rings"] == 2


@pytest.mark.skipif(docking.engine() != "conformer-mmff", reason="rdkit not installed")
def test_matched_pair_scores_above_a_fragment():
    matched = docking.score(SHP099, "SHP2", SHP2_POCKET)
    fragment = docking.score(ETHANOL, "SHP2", SHP2_POCKET)
    assert matched["score01"] > fragment["score01"] + 0.2
    assert matched["kcal_per_mol"] < fragment["kcal_per_mol"]


@pytest.mark.skipif(docking.engine() != "conformer-mmff", reason="rdkit not installed")
def test_wrong_target_scores_below_the_right_one():
    right = docking.score(SHP099, "SHP2", SHP2_POCKET)
    wrong_pocket = {"hydrophobicity": 0.3, "polarity": 0.7,
                    "mw_range": [450, 550], "radius_range": [5.0, 6.5],
                    "aromatic_rings_range": [3, 4], "reference_mw": 500}
    wrong = docking.score(SHP099, "EGFR", wrong_pocket)
    assert right["score01"] > wrong["score01"]


def test_score_is_deterministic():
    first = docking.score(SHP099, "SHP2", SHP2_POCKET)
    second = docking.score(SHP099, "SHP2", SHP2_POCKET)
    assert first["score01"] == second["score01"]


def test_bad_smiles_is_reported():
    result = docking.score("not-a-smiles", "SHP2", SHP2_POCKET)
    assert result["score01"] == 0.0
    assert "error" in result["detail"].lower() or result["engine"] == "unavailable"


# ------------------------------------------------------------------ vina adapter
def test_vina_availability_is_reported():
    assert vina_adapter.engine().startswith(("autodock-vina", "unavailable"))


def test_vina_requires_input_files(tmp_path):
    if not vina_adapter.available():
        pytest.skip("vina not installed")
    with pytest.raises(RuntimeError, match="PDBQT not found"):
        vina_adapter.dock(tmp_path / "missing.pdbqt", tmp_path / "missing2.pdbqt",
                          {"center": [0, 0, 0], "size": [20, 20, 20]})


def test_vina_calls_the_library(monkeypatch, tmp_path):
    """The adapter drives the real Vina API - checked with a stand-in module."""
    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.pdbqt"
    receptor.write_text("RECEPTOR", encoding="utf-8")
    ligand.write_text("LIGAND", encoding="utf-8")

    calls = {}

    class FakeVina:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def set_receptor(self, path):
            calls["receptor"] = path

        def set_ligand_from_file(self, path):
            calls["ligand"] = path

        def compute_vina_maps(self, center, box_size):
            calls["box"] = (center, box_size)

        def score(self):
            return [1.5, 2.0, 3.0]

        def dock(self, exhaustiveness=8, n_poses=5):
            calls["dock"] = (exhaustiveness, n_poses)

        def poses(self, n_poses=5):
            return "POSES"

        def energies(self, n_poses=5):
            return [[-9.4, 1.0, 2.0], [-8.9, 1.1, 2.1]]

    monkeypatch.setitem(sys.modules, "vina", types.SimpleNamespace(Vina=FakeVina))
    result = vina_adapter.dock(receptor, ligand,
                               {"center": [1, 2, 3], "size": [22, 22, 22]},
                               exhaustiveness=4)

    assert result["affinity_kcal_per_mol"] == -9.4
    assert result["n_poses"] == 2
    assert calls["box"] == ([1, 2, 3], [22, 22, 22])
    assert calls["dock"] == (4, 5)
