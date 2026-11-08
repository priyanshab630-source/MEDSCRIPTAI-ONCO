"""Dashboard endpoints: status, models, MC-dropout, compare, calibration."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient      # noqa: E402

from api.app import app                         # noqa: E402

client = TestClient(app)


def test_static_dir_resolves_portably():
    """`__file__` uses backslashes on Windows - the static dir must still resolve."""
    from api import app as app_module

    static = app_module.STATIC_DIR
    assert hasattr(static, "is_dir"), "STATIC_DIR should be a pathlib.Path"
    assert static.is_dir(), f"{static} is not a directory"
    assert "app.py" not in str(static)
    assert (static / "index.html").is_file()


def test_index_is_served_with_the_dashboard():
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    for marker in ("MEDSCRIPTAI", "panel-verdict", "panel-calibration",
                   "panel-models", "mc-smiles", "compare-input"):
        assert marker in html, marker


def test_health_reports_learned_components():
    body = client.get("/health").json()
    assert body["agents"] == 6
    assert "gnn" in body and "policy" in body
    assert body["engine"].startswith(("builtin", "langgraph"))


def test_models_endpoint_describes_checkpoints():
    body = client.get("/models").json()
    assert body["gnn"]["architecture"].startswith("2-layer")
    assert "REINFORCE" in body["policy"]["algorithm"]
    assert body["docking"]["engine"] in ("conformer-mmff", "descriptor-surrogate")


def test_tools_endpoint_lists_the_catalogue():
    body = client.get("/tools").json()
    assert len(body["tools"]) == 5
    assert body["active_policy"] in ("rl-policy", "deterministic")
    assert set(body["tools"]) == set(body["priority_order"])


def test_drugs_and_targets_feed_the_form():
    drugs = client.get("/drugs").json()
    targets = client.get("/targets").json()
    assert "SHP099" in drugs
    assert "SHP2" in targets


def test_datasets_endpoint_reports_provenance():
    body = client.get("/datasets").json()
    assert body["datasets"]
    esol = next(item for item in body["datasets"] if item["name"] == "esol")
    assert esol["available"] is True
    assert "Delaney" in esol["citation"]
    assert body["gnn_training"]["dataset"] in ("esol", "toy", None)


def test_analyze_returns_a_renderable_report():
    response = client.post("/analyze", json={"drug": "SHP099", "target": "SHP2",
                                             "disease": "KRAS-mutant NSCLC"})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in ("SUPPORTED", "CONTRADICTED", "UNCERTAIN", "ABSTAIN")
    assert len(body["uncertainty"]) == 6
    assert body["hypothesis"]["drug"] == "SHP099"
    assert body["tool_calls"], "the dashboard needs the tool transcript"
    assert body["trace"]


def test_analyze_rejects_unknown_compound_without_smiles():
    response = client.post("/analyze", json={"drug": "NothingKnown", "target": "SHP2"})
    assert response.status_code == 422


def test_uncertainty_endpoint():
    response = client.post("/uncertainty", json={"smiles": "CCO", "n_samples": 8})
    assert response.status_code == 200
    body = response.json()
    if not body["available"]:
        pytest.skip("no trained GNN checkpoint")
    assert 0.0 <= body["epistemic_uncertainty"] <= 1.0
    assert body["descriptors"]["backend"] == "rdkit"


def test_uncertainty_rejects_empty_smiles():
    assert client.post("/uncertainty", json={"smiles": "  "}).status_code == 422


def test_compare_returns_one_row_per_case():
    response = client.post("/compare", json={"cases": [
        {"drug": "SHP099", "target": "SHP2", "disease": "KRAS-mutant NSCLC"},
        {"drug": "SHP099", "target": "EGFR", "disease": "KRAS-mutant NSCLC"},
    ]})
    assert response.status_code == 200
    rows = response.json()["cases"]
    assert len(rows) == 2
    assert {(row["drug"], row["target"]) for row in rows} == {
        ("SHP099", "SHP2"), ("SHP099", "EGFR")}
    for row in rows:
        assert row["verdict"] in ("SUPPORTED", "CONTRADICTED", "UNCERTAIN", "ABSTAIN")
        assert 0.0 <= row["overall_uncertainty"] <= 1.0


def test_compare_rejects_an_empty_list():
    assert client.post("/compare", json={"cases": []}).status_code == 422


def test_calibration_endpoint_computes_metrics():
    response = client.get("/calibration?limit=4&refresh=true")
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    metrics = body["metrics"]
    for key in ("detection", "brier", "ece", "auroc_uncertainty", "abstention"):
        assert key in metrics
