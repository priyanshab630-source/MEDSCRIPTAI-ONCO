"""Interaction analysis on a hand-built complex with known geometry."""

import pytest

from src.tools import interactions


def _atom(record, name, resname, resnr, element, xyz):
    return {"record": record, "name": name, "resname": resname, "chain": "A",
            "resnr": str(resnr), "element": element, "xyz": xyz}


def write_complex(path, protein, ligand):
    """Write a PDB with exact column positions (the parser reads columns, not spaces)."""
    lines = []
    for index, atom in enumerate(protein + ligand, start=1):
        record = "HETATM" if atom["record"] == "HETATM" else "ATOM  "
        line = (f"{record}{index:5d} {atom['name']:<4s}"
                f" {atom['resname']:>3s} {atom['chain']}{atom['resnr']:>4s}    "
                f"{atom['xyz'][0]:8.3f}{atom['xyz'][1]:8.3f}{atom['xyz'][2]:8.3f}"
                f"  1.00  0.00          {atom['element']:>2s}")
        lines.append(line + "\n")
    path.write_text("".join(lines) + "END\n")
    return path


@pytest.fixture
def complex_pdb(tmp_path):
    protein = [
        _atom("ATOM", "OD1", "ASP", 51, "O", (0.0, 0.0, 0.0)),      # acceptor
        _atom("ATOM", "CG", "LEU", 60, "C", (0.0, 0.0, 8.0)),       # hydrophobic, far
        _atom("ATOM", "CD1", "LEU", 60, "C", (3.0, 0.0, 6.0)),      # hydrophobic, near
        _atom("ATOM", "NZ", "LYS", 70, "N", (0.0, 4.0, 0.0)),       # basic
    ]
    ligand = [
        _atom("HETATM", "N1", "UNL", 1, "N", (2.8, 0.0, 0.0)),      # 2.8 A to OD1
        _atom("HETATM", "C1", "UNL", 1, "C", (3.2, 0.0, 6.0)),      # 3.2 A to CD1
        _atom("HETATM", "O1", "UNL", 1, "O", (0.0, 3.6, 0.0)),      # 3.6 A to NZ
        _atom("HETATM", "C2", "UNL", 1, "C", (20.0, 20.0, 20.0)),   # far away
    ]
    return write_complex(tmp_path / "complex.pdb", protein, ligand)


def test_parses_protein_and_ligand(complex_pdb):
    parsed = interactions.parse_complex(str(complex_pdb))
    assert len(parsed["protein"]) == 4
    assert len(parsed["ligand"]) == 4
    assert parsed["ligand_id"].startswith("UNL")


def test_hydrogen_bond_detected(complex_pdb):
    report = interactions.analyse(str(complex_pdb))
    pairs = {(bond["ligand_atom"], bond["residue"]) for bond in report["hydrogen_bonds"]}
    assert ("N1", "ASP51") in pairs


def test_hydrophobic_contact_detected(complex_pdb):
    report = interactions.analyse(str(complex_pdb))
    assert any(contact["residue"] == "LEU60" for contact in report["hydrophobic_contacts"])


def test_salt_bridge_detected(complex_pdb):
    report = interactions.analyse(str(complex_pdb))
    assert any(bridge["residue"] == "LYS70" for bridge in report["salt_bridges"])


def test_distant_atoms_are_not_contacts(complex_pdb):
    report = interactions.analyse(str(complex_pdb))
    names = {bond["ligand_atom"] for bond in report["hydrogen_bonds"]}
    names |= {contact["ligand_atom"] for contact in report["hydrophobic_contacts"]}
    assert "C2" not in names


def test_counts_match_the_lists(complex_pdb):
    report = interactions.analyse(str(complex_pdb))
    assert report["counts"]["hydrogen_bonds"] == len(report["hydrogen_bonds"])
    assert report["counts"]["binding_site_residues"] == len(report["binding_site_residues"])


def test_missing_ligand_is_reported(tmp_path):
    protein_only = write_complex(tmp_path / "protein.pdb",
                                 [_atom("ATOM", "CA", "ALA", 1, "C", (0.0, 0.0, 0.0))], [])
    with pytest.raises(ValueError, match="no HETATM ligand"):
        interactions.parse_complex(str(protein_only))
