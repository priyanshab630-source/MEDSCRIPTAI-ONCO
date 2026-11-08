"""The fetch/verify CLI: manifest listing, checksum verification, Kaggle guidance."""

import subprocess
import sys


def _run(*args):
    return subprocess.run([sys.executable, "scripts/fetch_datasets.py", *args],
                          capture_output=True, text=True, timeout=120)


def test_list_shows_every_dataset_with_citation():
    result = _run("--list")
    assert result.returncode == 0
    assert "esol" in result.stdout
    assert "Delaney" in result.stdout
    assert "Kaggle" in result.stdout


def test_verify_passes_for_bundled_data():
    result = _run("--verify")
    assert result.returncode == 0, result.stdout
    assert "[ok  ] esol" in result.stdout


def test_verify_flags_a_tampered_file(tmp_path, monkeypatch):
    import hashlib

    from src.tools import datasets

    entry = datasets.get("esol")
    copy = tmp_path / "esol.csv"
    copy.write_text("smiles,target\nCC,1\n", encoding="utf-8")
    monkeypatch.setitem(entry, "file", str(copy))
    monkeypatch.setitem(entry, "sha256", hashlib.sha256(b"different").hexdigest())

    report = datasets.verify("esol")
    assert report["ok"] is False
    assert "mismatch" in report["reason"]


def test_kaggle_without_cli_explains_the_token():
    result = _run("--kaggle", "owner/dataset", "--name", "x")
    if result.returncode == 0:
        pytest.skip("kaggle CLI is installed here")
    assert "kaggle.json" in result.stdout or "kaggle.json" in result.stderr
