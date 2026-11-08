"""Training entry point: dataset wiring, splits and metadata (tiny run)."""

import json

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional extra")
pytest.importorskip("rdkit", reason="rdkit is an optional extra")

from training import train_gnn  # noqa: E402


def test_toy_dataset_trains_and_writes_metadata(tmp_path):
    out = tmp_path / "gnn.pt"
    code = train_gnn.main(["--dataset", "toy", "--epochs", "2",
                           "--out", str(out)])
    assert code == 0
    assert out.exists()

    metadata = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["dataset"] == "toy"
    assert metadata["task"] == "regression"
    assert "mae" in metadata["test"]


def test_real_dataset_records_provenance(tmp_path):
    """ESOL is bundled, so this runs offline and must carry its citation."""
    out = tmp_path / "gnn.pt"
    code = train_gnn.main(["--dataset", "esol", "--epochs", "1",
                           "--out", str(out)])
    assert code == 0

    metadata = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["dataset"] == "esol"
    assert metadata["molecules"] > 1000
    assert metadata["citation"].startswith("Delaney")
    assert len(metadata["dataset_sha256"]) == 64
    assert metadata["split"]["train"] > metadata["split"]["test"]


def test_missing_dataset_fails_loudly(tmp_path):
    with pytest.raises(SystemExit):
        train_gnn.main(["--dataset", "qm7", "--epochs", "1",
                        "--out", str(tmp_path / "x.pt")])
