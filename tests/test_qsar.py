"""Descriptor layer: RDKit when installed, graph estimates otherwise."""

import pytest

from src.tools import qsar

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_backend_is_reported():
    assert qsar.backend() in ("rdkit", "graph-estimates")


@pytest.mark.skipif(qsar.backend() != "rdkit", reason="rdkit not installed")
def test_rdkit_matches_known_values():
    descriptors = qsar.compute(ASPIRIN)
    assert descriptors["backend"] == "rdkit"
    assert descriptors["mw"] == pytest.approx(180.16, abs=0.5)   # aspirin
    assert descriptors["tpsa"] == pytest.approx(63.6, abs=1.0)
    assert descriptors["hbd"] == 1
    assert descriptors["rings"] == 1
    assert 0.0 < descriptors["qed"] < 1.0


@pytest.mark.skipif(qsar.backend() != "rdkit", reason="rdkit not installed")
def test_rdkit_rejects_bad_smiles():
    assert "error" in qsar.compute("this-is-not-smiles")


def test_graph_fallback_still_works():
    """The pure-Python path must keep working for installs without RDKit."""
    descriptors = qsar._graph_descriptors(ASPIRIN)
    assert descriptors["backend"] == "graph-estimates"
    assert 170 < descriptors["mw"] < 200
    assert descriptors["hbd"] == 1


def test_profile_scoring():
    descriptors = qsar.compute(ASPIRIN)
    score, detail = qsar.score_against_profile(
        descriptors, {"mw": [150, 200], "logp": [0, 3], "tpsa": [50, 80]})
    assert score == 1.0
    assert "mw=" in detail
