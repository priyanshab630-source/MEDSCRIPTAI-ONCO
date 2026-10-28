"""The fine-tuning scripts actually run (tiny mode, CPU, seconds)."""

import json

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional extra")
pytest.importorskip("transformers", reason="transformers is an optional extra")

from training import build_dataset, dpo_train, sft_train  # noqa: E402

REPORT = {
    "verdict": "ABSTAIN",
    "rationale": "Abstaining: the evidence base cannot resolve this claim.",
    "overall_uncertainty": 0.92,
    "dominant_dimension": "evidence_coverage",
    "support_score": 0.0,
    "uncertainty": {"evidence_coverage": 1.0, "model_reliability": 0.45},
    "abstain_reasons": ["no assay-level evidence"],
    "evidence": [{"source": "literature", "claim": "nothing relevant"}],
    "computations": [],
    "hypothesis": {"drug": "CompoundX", "target": "SHP2", "disease": ""},
}


def _write_dataset(tmp_path):
    build_dataset.build([REPORT, REPORT], tmp_path)
    return tmp_path


def test_sft_tiny_run(tmp_path):
    out = tmp_path / "model"
    code = sft_train.main(["--tiny", "--data", str(tmp_path / "sft.jsonl"),
                           "--out", str(out), "--max-steps", "1"])
    assert code == 0
    assert (out / "config.json").exists()


def test_dpo_tiny_run(tmp_path):
    out = tmp_path / "model"
    code = dpo_train.main(["--tiny", "--data", str(tmp_path / "dpo.jsonl"),
                           "--out", str(out), "--max-steps", "1"])
    assert code == 0
    assert (out / "config.json").exists()


@pytest.fixture(autouse=True)
def dataset(tmp_path, monkeypatch):
    _write_dataset(tmp_path)
    return tmp_path
