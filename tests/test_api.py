"""API surface: health, drug table, analysis, validation error."""

import pytest

pytest.importorskip("fastapi")

try:                                            # starlette's TestClient needs httpx2
    from fastapi.testclient import TestClient
except Exception as exc:                        # noqa: BLE001
    pytest.skip(f"TestClient unavailable: {exc}", allow_module_level=True)

from api.app import app                         # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["agents"] == 6


def test_drug_table():
    drugs = client.get("/drugs").json()
    assert "SHP099" in drugs


def test_analyze_known_compound():
    response = client.post("/analyze", json={"drug": "SHP099", "target": "SHP2",
                                             "disease": "KRAS-mutant NSCLC"})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in ("SUPPORTED", "UNCERTAIN", "CONTRADICTED", "ABSTAIN")
    assert "uncertainty" in body


def test_analyze_unknown_compound_needs_smiles():
    response = client.post("/analyze", json={"drug": "NothingKnown", "target": "SHP2"})
    assert response.status_code == 422
