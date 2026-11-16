"""The Docking tab must render in every state, and never call a function that is gone.

This exists because a refactor deleted `renderDocking` while leaving its callers, and
the tab silently sat on "Loading…" - a failure the API tests could not see.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

node = shutil.which("node")
HARNESS = Path("tests/docking_tab_check.js")


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_docking_tab_renders_in_all_three_states():
    result = subprocess.run([node, str(HARNESS)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert result.stdout.count("PASS") >= 8
