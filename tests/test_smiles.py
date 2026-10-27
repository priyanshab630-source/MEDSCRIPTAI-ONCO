"""Molecular graph reader: atoms, rings, implicit hydrogens."""

from src.tools import qsar, smiles


def test_aspirin_graph():
    mol = smiles.parse("CC(=O)Oc1ccccc1C(=O)O")
    assert len(mol["atoms"]) == 13
    assert smiles.ring_count(mol) == 1
    assert smiles.heavy_atom_count(mol) == 13


def test_branch_and_ring_digits():
    mol = smiles.parse("c1ccccc1")
    assert len(mol["atoms"]) == 6
    assert len(mol["bonds"]) == 6
    assert smiles.ring_count(mol) == 1


def test_implicit_hydrogens_on_methyl():
    mol = smiles.parse("CC")
    assert smiles.implicit_hydrogens(mol) == [3, 3]


def test_two_letter_element():
    mol = smiles.parse("CCl")
    assert [a["symbol"] for a in mol["atoms"]] == ["C", "Cl"]


def test_empty_string_is_safe():
    assert smiles.parse("") == {"atoms": [], "bonds": []}
    assert "error" in qsar.compute("")


def test_descriptors_are_sane():
    descriptors = qsar.compute("CC(=O)Oc1ccccc1C(=O)O")
    assert 170 < descriptors["mw"] < 200          # aspirin is 180.16
    assert descriptors["hbd"] == 1
    assert descriptors["hba"] == 4
    assert descriptors["rings"] == 1
    assert descriptors["drug_like"] is True
