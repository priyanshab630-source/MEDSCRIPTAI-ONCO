"""The trained GNN checkpoint, when torch and the weights are present."""

import pytest

from src.tools import gnn, gnn_torch

torch = pytest.importorskip("torch", reason="torch is an optional extra")

SHP099 = "Nc1nc(N2CCNCC2)c2cc(Cl)ccc2n1"


def test_checkpoint_loads():
    model = gnn_torch.load()
    if model is None:
        pytest.skip("no trained checkpoint - run training/train_gnn.py")
    assert model is not None


def test_embedding_is_learned_and_fixed_size():
    model = gnn_torch.load()
    if model is None:
        pytest.skip("no trained checkpoint")
    vector = gnn_torch.embed(SHP099, model)
    assert len(vector) == gnn_torch.HIDDEN
    assert any(abs(v) > 0 for v in vector)


def test_predict_property_is_close_to_the_proxy_target():
    model = gnn_torch.load()
    if model is None:
        pytest.skip("no trained checkpoint")
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    predicted = gnn_torch.predict_property("CC(=O)Oc1ccccc1C(=O)O", model)
    actual = Descriptors.MolLogP(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    # A 33-molecule toy task: assert the right ballpark, not accuracy.
    assert abs(predicted - actual) < 3.0


def test_report_says_which_gnn_answered():
    _, detail = gnn.predict_binding(SHP099, {"mw": [280, 500]})
    if gnn_torch.load() is None:
        assert "untrained" in detail
    else:
        assert "trained checkpoint" in detail
