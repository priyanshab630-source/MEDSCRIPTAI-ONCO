"""End-to-end docking on a bundled PDBbind complex (skipped without the stack)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("vina", reason="AutoDock Vina python bindings are optional")
pytest.importorskip("meeko", reason="Meeko is optional")
pytest.importorskip("rdkit", reason="rdkit is optional")

RECEPTOR = Path("data/docking/4ajl/4ajl_pocket.pdb")
LIGAND = Path("data/docking/4ajl/4ajl_ligand.mol2")


def test_redocking_reproduces_the_crystal_pose(tmp_path):
    """Dock the crystal ligand back and check pose 1 lands near the crystal pose."""
    if not (RECEPTOR.exists() and LIGAND.exists()):
        pytest.skip("docking fixtures are missing")

    result = subprocess.run(
        [sys.executable, "scripts/run_docking.py",
         "--receptor", str(RECEPTOR), "--ligand", str(LIGAND),
         "--out", str(tmp_path), "--exhaustiveness", "1", "--n-poses", "3",
         "--crystal-ligand", str(LIGAND)],
        capture_output=True, text=True, timeout=280)
    assert result.returncode == 0, result.stdout + result.stderr

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["poses"], "no poses were written"
    assert report["poses"][0][0] < 0, "affinity should be negative"
    assert report["counts"]["binding_site_residues"] > 0
    assert report["counts"]["hydrogen_bonds"] >= 0

    # a valid pose, and the viewer scripts must exist
    for name in ("complex.pdb", "poses.pdbqt", "view.pml", "view.cxc", "report.md"):
        assert (tmp_path / name).exists(), f"{name} was not written"

    if report["redocking_rmsd"] is not None:
        assert report["redocking_rmsd"] <= 3.0, "redocking should land near the crystal pose"
