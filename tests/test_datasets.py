"""Dataset manifest, loading, validation, splitting and provenance."""

import pytest

from src.tools import datasets


def test_manifest_entries_have_provenance():
    entries = datasets.entries()
    assert len(entries) >= 2
    for entry in entries:
        for field in ("name", "file", "task", "source", "citation", "license"):
            assert entry.get(field), f"{entry.get('name')} is missing {field}"
        assert entry["task"] in ("regression", "classification")


def test_bundled_datasets_verify():
    for entry in datasets.entries():
        if entry.get("downloaded_at") is None:
            continue
        report = datasets.verify(entry["name"])
        assert report["ok"], report["reason"]


def test_esol_loads_with_real_values():
    rows = datasets.load("esol")
    assert len(rows) > 1000
    assert all(-12 < row["target"] < 2 for row in rows[:50])
    assert all(row["smiles"] for row in rows[:50])


def test_bace_is_classification():
    entry = datasets.get("bace")
    assert entry["task"] == "classification"
    rows = datasets.load("bace")
    assert len(rows) > 1000
    assert {row["label"] for row in rows} <= {0, 1}


def test_split_is_deterministic_and_disjoint():
    rows = datasets.load("esol")[:200]
    train_a, val_a, test_a = datasets.split(rows, seed=13)
    train_b, val_b, test_b = datasets.split(rows, seed=13)
    assert train_a == train_b and val_a == val_b and test_a == test_b

    train, val, test = train_a, val_a, test_a
    assert len(train) + len(val) + len(test) == len(rows)
    seen = {row["smiles"] for row in train}
    assert not (seen & {row["smiles"] for row in val})
    assert not (seen & {row["smiles"] for row in test})


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError):
        datasets.get("does-not-exist")


def test_missing_file_reports_how_to_fetch(tmp_path, monkeypatch):
    entry = datasets.get("esol")
    monkeypatch.setitem(entry, "file", str(tmp_path / "absent.csv"))
    report = datasets.verify("esol")
    assert report["ok"] is False
    assert "fetch_datasets" in report["hint"]


def test_summary_feeds_the_api():
    summary = datasets.summary()
    esol = next(item for item in summary if item["name"] == "esol")
    assert esol["available"] is True
    assert esol["loaded_rows"] > 1000
    assert esol["citation"].startswith("Delaney")
    assert len(esol["sha256"]) == 64
