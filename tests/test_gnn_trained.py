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


def test_prediction_matches_the_checkpoints_own_task():
    """The head predicts whatever the checkpoint was trained on - not MolLogP."""
    model = gnn_torch.load()
    if model is None:
        pytest.skip("no trained checkpoint")

    info = gnn_torch.metadata()
    predicted = gnn_torch.predict_property("CC(=O)Oc1ccccc1C(=O)O", model)
    assert isinstance(predicted, float)

    if info.get("task") == "regression":
        mean = info.get("target_mean", 0.0)
        std = info.get("target_std", 1.0)
        # a prediction outside ±5 training spreads means the model is broken
        assert abs(predicted - mean) < 5 * std
    assert info.get("dataset"), "metadata must record the training dataset"


def test_report_says_which_gnn_answered():
    _, detail = gnn.predict_binding(SHP099, {"mw": [280, 500]})
    if gnn_torch.load() is None:
        assert "untrained" in detail
    else:
        assert "trained checkpoint" in detail
