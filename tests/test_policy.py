"""The REINFORCE tool-selection policy."""

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional extra")
from src.tools import policy  # noqa: E402


def test_feature_vector_shape():
    features = policy.encode([], has_smiles=True, max_steps=4)
    assert len(features) == policy.FEATURE_DIM
    assert features[0] == 1.0


def test_features_track_progress():
    transcript = [
        {"tool": "docking", "ok": True, "score": 0.8, "observation": "x"},
        {"tool": "kg_lookup", "ok": True, "observation": "y"},
    ]
    features = policy.encode(transcript, has_smiles=True, max_steps=4)
    assert features[3] == 1.0            # has a computational signal
    assert features[4] == 1.0            # has an evidence signal
    assert features[5] == 0.8            # mean computational score
    assert features[6 + policy.TOOLS.index("docking")] == 1.0


def test_mask_excludes_used_tools():
    mask = policy.mask_for([{"tool": "docking", "ok": True, "observation": ""}])
    assert mask[policy.TOOLS.index("docking")] == 0.0
    assert sum(mask) == len(policy.TOOLS) - 1


def test_trained_policy_selects_an_unused_tool():
    model = policy.load()
    if model is None:
        pytest.skip("no trained policy - run training/train_policy.py")
    choice = policy.select([], has_smiles=True, max_steps=4, model=model)
    assert choice in policy.TOOLS

    used = [{"tool": choice, "ok": True, "observation": ""}]
    second = policy.select(used, has_smiles=True, max_steps=4, model=model)
    assert second != choice


def test_policy_metadata_describes_the_run():
    info = policy.metadata()
    if not info:
        pytest.skip("no trained policy")
    assert info["episodes"] > 0
    assert info["tools"] == policy.TOOLS


def test_no_checkpoint_means_no_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "CHECKPOINT", tmp_path / "missing.pt")
    assert policy.load() is None
