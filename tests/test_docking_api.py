"""Docking endpoints, scene builder and PLIP report parsing."""

import json
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402
from src.tools import plip_adapter, scene  # noqa: E402

client = TestClient(app)


def test_docking_endpoint_reports_the_run():
    body = client.get("/docking").json()
    if not body.get("available"):
        pytest.skip("no docking run in this checkout")
    assert body["poses"]
    assert body["redocking_rmsd"] is None or body["redocking_rmsd"] < 3.0
    assert "own_analysis" in body and "plip" in body


def test_scene_has_ligand_bonds_and_hbonds():
    body = client.get("/docking").json()
    if not body.get("available"):
        pytest.skip("no docking run in this checkout")
    geometry = client.get("/docking/scene").json()
    assert geometry["counts"]["ligand_atoms"] > 10
    assert geometry["counts"]["bonds"] > 10
    assert geometry["counts"]["protein_atoms"] > 50
    assert len(geometry["ligand"]) == geometry["counts"]["ligand_atoms"]


def test_printable_view_renders_svg():
    response = client.get("/docking/view")
    if response.status_code == 404:
        pytest.skip("no docking run in this checkout")
    assert response.status_code == 200
    assert "<svg" in response.text
    assert "PLIP" in response.text


def test_plip_parser_reads_the_nested_interactions(tmp_path):
    """PLIP nests lists under bindingsite/interactions - the parser must know that."""
    xml = """<report><bindingsite>
      <identifiers><longname>UNL</longname><smiles>CC</smiles></identifiers>
      <interactions>
        <hydrogen_bonds>
          <site><resnr>51</resnr><restype>ASP</restype><dist>1.94</dist>
            <donortype>Nam</donortype><acceptortype>O3</acceptortype></site>
        </hydrogen_bonds>
        <hydrophobic_interactions>
          <site><resnr>30</resnr><restype>VAL</restype><dist>3.48</dist></site>
          <site><resnr>52</resnr><restype>VAL</restype><dist>3.67</dist></site>
        </hydrophobic_interactions>
      </interactions>
    </bindingsite></report>"""
    path = tmp_path / "report.xml"
    path.write_text(xml)

    parsed = plip_adapter.parse_report(path)
    assert parsed["counts"]["hydrogen_bonds"] == 1
    assert parsed["counts"]["hydrophobic_contacts"] == 2
    assert parsed["hydrogen_bonds"][0]["residue"] == "ASP51"
    assert parsed["hydrogen_bonds"][0]["distance"] == 1.94
    # residues are derived from the interactions when PLIP leaves bs_residues empty
    assert "ASP51" in parsed["binding_site_residues"]


def test_scene_requires_a_ligand(tmp_path):
    protein_only = tmp_path / "protein.pdb"
    protein_only.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    with pytest.raises(ValueError, match="no ligand"):
        scene.build_scene(protein_only, {})
