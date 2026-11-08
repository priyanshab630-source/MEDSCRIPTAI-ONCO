"""Run the dashboard's own render functions against a real report payload.

The browser is not available in CI, so this harness stubs the DOM and executes the
page's JavaScript with Node. It catches the failure that actually bites: a render
function that throws, or a tab that comes back empty, because a field was renamed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
node = shutil.which("node")

HARNESS = Path("tests/render_check.js")


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_render_functions_produce_markup(tmp_path):
    from fastapi.testclient import TestClient

    from api.app import app

    report = TestClient(app).post("/analyze", json={
        "drug": "SHP099", "target": "SHP2", "disease": "KRAS-mutant NSCLC",
    }).json()
    assert report["verdict"], "the pipeline must return a verdict first"

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    script_path = tmp_path / "render_check.js"
    script_path.write_text(HARNESS.read_text(encoding="utf-8"))

    result = subprocess.run([node, str(script_path), str(report_path)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
