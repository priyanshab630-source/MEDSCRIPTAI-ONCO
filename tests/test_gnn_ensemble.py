"""MC-dropout uncertainty from the trained GNN."""

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional extra")
from src.tools import gnn_torch  # noqa: E402

IN_DISTRIBUTION = "c1ccccc1"          # benzene: in the training library
OUT_OF_DISTRIBUTION = "CCO"           # ethanol: also in the library, but simple


def _model():
    model = gnn_torch.load()
    if model is None:
        pytest.skip("no trained checkpoint - run training/train_gnn.py")
    return model


def test_distribution_returns_mean_and_spread():
    model = _model()
    mean, std = gnn_torch.predict_distribution(IN_DISTRIBUTION, n_samples=10, model=model)
    assert isinstance(mean, float)
    assert std >= 0.0


def test_mc_dropout_is_deterministic():
    model = _model()
    first = gnn_torch.predict_distribution(IN_DISTRIBUTION, n_samples=10, model=model)
    second = gnn_torch.predict_distribution(IN_DISTRIBUTION, n_samples=10, model=model)
    assert first == second


def test_epistemic_uncertainty_is_bounded():
    value = gnn_torch.epistemic_uncertainty(IN_DISTRIBUTION, n_samples=10)
    assert 0.0 <= value <= 1.0


def test_uncertainty_varies_between_molecules():
    """If every molecule had the same spread the measure would be worthless."""
    values = {gnn_torch.epistemic_uncertainty(smiles, n_samples=10)
              for smiles in (IN_DISTRIBUTION, OUT_OF_DISTRIBUTION, "CCCCCCCC")}
    assert len(values) > 1


def test_uncertainty_feeds_the_vector():
    from src.core import uncertainty as unc
    from src.core.schemas import EvidenceItem

    evidence = [EvidenceItem("literature", "claim", 1, 0.8)]
    measured = unc.build(evidence, [], model_reliability_measured=0.12)
    fallback = unc.build(evidence, [], model_reliability_measured=None)
    assert measured.model_reliability == 0.12
    assert fallback.model_reliability == unc.MODEL_RELIABILITY_PENALTY
