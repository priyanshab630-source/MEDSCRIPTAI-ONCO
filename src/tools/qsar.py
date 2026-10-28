"""QSAR layer.

Two backends, same dict of descriptors:

  * **RDKit** (default when installed) - MolWt, MolLogP, HBD/HBA, TPSA, ring and
    rotatable-bond counts, QED. This is the real thing; install with `pip install rdkit`.
  * **graph estimates** - the pure-Python fallback from `src/tools/smiles.py`, so
    the prototype still runs with no chemistry toolchain.

`backend()` tells you which one answered, and every report records it.
"""

from src.tools import smiles as smiles_tool

ATOMIC_MASS = smiles_tool.ATOMIC_MASS

# --- fallback: rough fragment contributions, relative comparison only ---------
LOGP_CONTRIB = {"C": 0.35, "N": -0.55, "O": -0.65, "S": 0.25,
                "F": 0.35, "Cl": 0.55, "Br": 0.75, "I": 0.95, "P": 0.1, "B": 0.1}
TPSA_FRAGMENTS = {
    ("O", 0, False): 9.2, ("O", 1, False): 20.2,
    ("N", 0, False): 3.2, ("N", 1, False): 12.0, ("N", 2, False): 26.0,
    ("N", 0, True): 12.9, ("N", 1, True): 15.8,
    ("S", 0, False): 25.3,
}
DOUBLE_BONDED_O = 17.1


def backend() -> str:
    """'rdkit' when the chemistry toolkit is installed, else 'graph-estimates'."""
    try:
        import rdkit  # noqa: F401

        return "rdkit"
    except ImportError:
        return "graph-estimates"


# ---------------------------------------------------------------- RDKit backend
def _rdkit_descriptors(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    return {
        "smiles": smiles,
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "mw": round(Descriptors.MolWt(molecule), 2),
        "logp": round(Descriptors.MolLogP(molecule), 2),
        "hbd": Lipinski.NumHDonors(molecule),
        "hba": Lipinski.NumHAcceptors(molecule),
        "rings": rdMolDescriptors.CalcNumRings(molecule),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(molecule),
        "aromatic_atoms": sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic()),
        "tpsa": round(rdMolDescriptors.CalcTPSA(molecule), 1),
        "qed": round(QED.qed(molecule), 3),
        "backend": "rdkit",
    }


# ------------------------------------------------------- pure-Python fallback
def molecular_weight(mol: dict) -> float:
    hydrogens = smiles_tool.implicit_hydrogens(mol)
    total = 0.0
    for atom, h_count in zip(mol["atoms"], hydrogens):
        total += ATOMIC_MASS.get(atom["symbol"].capitalize(), 12.0)
        total += h_count * ATOMIC_MASS["H"]
    return round(total, 2)


def h_bond_donors(mol: dict) -> int:
    hydrogens = smiles_tool.implicit_hydrogens(mol)
    return sum(1 for atom, h in zip(mol["atoms"], hydrogens)
               if atom["symbol"] in ("N", "O") and h > 0)


def h_bond_acceptors(mol: dict) -> int:
    return sum(1 for a in mol["atoms"] if a["symbol"] in ("N", "O"))


def rotatable_bonds(mol: dict) -> int:
    ring_flags = smiles_tool.ring_bond_flags(mol)
    adj = smiles_tool.adjacency(mol)
    degree = {i: len(adj[i]) for i in adj}
    return sum(
        1 for (a, b, order), in_ring in zip(mol["bonds"], ring_flags)
        if order == 1.0 and not in_ring and degree[a] > 1 and degree[b] > 1
    )


def logp(mol: dict) -> float:
    return round(sum(LOGP_CONTRIB.get(a["symbol"].capitalize(), 0.2)
                     for a in mol["atoms"]), 2)


def tpsa(mol: dict) -> float:
    hydrogens = smiles_tool.implicit_hydrogens(mol)
    adj = smiles_tool.adjacency(mol)
    total = 0.0
    for i, atom in enumerate(mol["atoms"]):
        symbol = atom["symbol"].capitalize()
        if symbol == "O":
            double_bonded = any(order >= 2 for _, order in adj[i])
            total += DOUBLE_BONDED_O if double_bonded else TPSA_FRAGMENTS[("O", 0, False)]
        elif symbol in ("N", "S"):
            key = (symbol, min(hydrogens[i], 2), atom["aromatic"])
            total += TPSA_FRAGMENTS.get(key, TPSA_FRAGMENTS.get((symbol, 0, False), 0.0))
    return round(total, 1)


def _graph_descriptors(smiles: str):
    mol = smiles_tool.parse(smiles)
    if not mol["atoms"]:
        return None
    return {
        "smiles": smiles,
        "heavy_atoms": smiles_tool.heavy_atom_count(mol),
        "mw": molecular_weight(mol),
        "logp": logp(mol),
        "hbd": h_bond_donors(mol),
        "hba": h_bond_acceptors(mol),
        "rings": smiles_tool.ring_count(mol),
        "rotatable_bonds": rotatable_bonds(mol),
        "aromatic_atoms": sum(1 for a in mol["atoms"] if a["aromatic"]),
        "tpsa": tpsa(mol),
        "qed": None,
        "backend": "graph-estimates",
    }


# ------------------------------------------------------------------- public API
def compute(smiles: str) -> dict:
    """Full descriptor set for one SMILES string."""
    descriptors = None
    if backend() == "rdkit":
        descriptors = _rdkit_descriptors(smiles)
        if descriptors is None:
            return {"error": "RDKit could not parse this SMILES", "smiles": smiles}

    if descriptors is None:
        descriptors = _graph_descriptors(smiles)
    if descriptors is None:
        return {"error": "could not parse SMILES", "smiles": smiles}

    passed = sum([
        descriptors["mw"] <= 500,
        descriptors["logp"] <= 5,
        descriptors["hbd"] <= 5,
        descriptors["hba"] <= 10,
    ])
    descriptors["lipinski_passed"] = passed
    descriptors["drug_like"] = passed >= 3
    return descriptors


def score_against_profile(descriptors: dict, profile: dict) -> tuple:
    """Compare descriptors with a target's reference window -> (score, detail)."""
    if "error" in descriptors:
        return 0.0, descriptors["error"]

    checks, hits = [], 0
    for field in ("mw", "logp", "hbd", "hba", "tpsa"):
        window = profile.get(field)
        if not window:
            continue
        low, high = window
        value = descriptors.get(field, 0.0)
        ok = low <= value <= high
        hits += 1 if ok else 0
        checks.append(f"{field}={value} ({'in' if ok else 'outside'} {low:g}-{high:g})")

    score = hits / len(checks) if checks else 0.5
    return round(score, 4), "; ".join(checks)
