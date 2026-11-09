"""ChEMBL fetcher: target resolution, activity paging, SMILES joining, provenance.

The HTTP layer is replaced with the recorded responses in `data/fixtures/chembl_*`,
so the client logic is tested without touching ebi.ac.uk.
"""

import csv
import json
from pathlib import Path

import pytest

from scripts import fetch_chembl

FIXTURES = Path("data/fixtures")


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _router(url, params=None, timeout=None, headers=None):
    params = params or {}
    if "target.json" in url or "target/" in url:
        return FakeResponse(json.loads((FIXTURES / "chembl_targets.json").read_text()))
    if "activity.json" in url:
        return FakeResponse(json.loads((FIXTURES / "chembl_activities.json").read_text()))
    if "molecule.json" in url:
        return FakeResponse(json.loads((FIXTURES / "chembl_molecules.json").read_text()))
    raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(fetch_chembl.requests, "get", _router)
    return monkeypatch


def test_target_resolution(offline):
    target = fetch_chembl.resolve_target("PTPN11")
    assert target["target_chembl_id"] == "CHEMBL3351120"
    assert "phosphatase" in target["pref_name"]


def test_activities_drop_rows_without_values(offline):
    rows = fetch_chembl.fetch_activities("CHEMBL3351120", "IC50", limit=100)
    assert len(rows) == 2                     # the null-value row is dropped
    assert {row["standard_value"] for row in rows} == {"71.0", "11.0"}


def test_smiles_join(offline):
    rows = fetch_chembl.fetch_activities("CHEMBL3351120", "IC50", limit=100)
    joined = fetch_chembl.attach_smiles(rows)
    assert len(joined) == 2
    assert all(row["smiles"] for row in joined)


def test_end_to_end_writes_csv_and_registers(offline, tmp_path):
    """Registers into a throwaway copy of the manifest, never the real one."""
    import shutil

    from src.tools import datasets

    manifest_copy = tmp_path / "manifest.json"
    shutil.copyfile(datasets.MANIFEST, manifest_copy)
    offline.setattr(datasets, "MANIFEST", manifest_copy)
    offline.setattr(fetch_chembl, "DESTINATION", tmp_path)
    datasets.reload()
    try:
        code = fetch_chembl.main(["--target", "PTPN11", "--name", "chembl_test",
                                  "--limit", "50"])
        assert code == 0

        written = tmp_path / "chembl_test.csv"
        assert written.exists()
        with open(written, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["pchembl_value"]
        assert rows[0]["smiles"]

        entry = next(e for e in datasets.entries() if e["name"] == "chembl_test")
        assert entry["source"].startswith("ChEMBL API")
        assert entry["target_column"] == "pchembl_value"
        assert "CC BY-SA" in entry["license"]
        assert len(entry["sha256"]) == 64
    finally:
        datasets.reload()                     # back to the real manifest
