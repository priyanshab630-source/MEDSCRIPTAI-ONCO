"""QSAR layer: Lipinski-style descriptors from the molecular graph.

Fast graph-based estimates so the pipeline runs with no chemistry toolchain.
Good enough to ask "is this compound drug-like, and does it fit the target's
reference profile?". For production, swap `compute()` for RDKit descriptors -
the callers only need the same dict keys.
"""

from src.tools import smiles as smiles_tool

ATOMIC_MASS = smiles_tool.ATOMIC_MASS

# Rough fragment contributions, used for relative comparison only.
LOGP_CONTRIB = {"C": 0.35, "N": -0.55, "O": -0.65, "S": 0.25,
                "F": 0.35, "Cl": 0.55, "Br": 0.75, "I": 0.95, "P": 0.1, "B": 0.1}

# Ertl-style polar surface area fragments, keyed by (element, hydrogens, aromatic).
TPSA_FRAGMENTS = {
    ("O", 0, False): 9.2,      # ether / ester oxygen
    ("O", 1, False): 20.2,     # hydroxyl
    ("N", 0, False): 3.2,      # tertiary amine
    ("N", 1, False): 12.0,     # secondary amine
    ("N", 2, False): 26.0,     # primary amine
    ("N", 0, True): 12.9,      # pyridine-type nitrogen
    ("N", 1, True): 15.8,      # pyrrole-type [nH]
    ("S", 0, False): 25.3,
}
DOUBLE_BONDED_O = 17.1         # carbonyl oxygen


def molecular_weight(mol: dict) -> float:
    hydrogens = smiles_tool.implicit_hydrogens(mol)
    total = 0.0
    for atom, h_count in zip(mol["atoms"], hydrogens):
        total += ATOMIC_MASS.get(atom["symbol"].capitalize(), 12.0)
        total += h_count * ATOMIC_MASS["H"]
    return round(total, 2)


def h_bond_donors(mol: dict) -> int:
    hydrogens = smiles_tool.implicit_hydrogens(mol)
    return sum(
        1 for atom, h in zip(mol["atoms"], hydrogens)
        if atom["symbol"] in ("N", "O") and h > 0
    )


def h_bond_acceptors(mol: dict) -> int:
    return sum(1 for a in mol["atoms"] if a["symbol"] in ("N", "O"))


def rotatable_bonds(mol: dict) -> int:
    """Single, acyclic bonds between two heavy atoms that both have >1 neighbour."""
    ring_flags = smiles_tool.ring_bond_flags(mol)
    adj = smiles_tool.adjacency(mol)
    degree = {i: len(adj[i]) for i in adj}
    count = 0
    for (a, b, order), in_ring in zip(mol["bonds"], ring_flags):
        if order == 1.0 and not in_ring and degree[a] > 1 and degree[b] > 1:
            count += 1
    return count


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
            key = ("O", 0, False) if not double_bonded else None
            total += DOUBLE_BONDED_O if key is None else TPSA_FRAGMENTS[key]
        elif symbol in ("N", "S"):
            key = (symbol, min(hydrogens[i], 2), atom["aromatic"])
            total += TPSA_FRAGMENTS.get(key, TPSA_FRAGMENTS.get((symbol, 0, False), 0.0))
    return round(total, 1)


def aromatic_atoms(mol: dict) -> int:
    return sum(1 for a in mol["atoms"] if a["aromatic"])


def compute(smiles: str) -> dict:
    """Full descriptor set for one SMILES string."""
    mol = smiles_tool.parse(smiles)
    if not mol["atoms"]:
        return {"error": "could not parse SMILES", "smiles": smiles}

    descriptors = {
        "smiles": smiles,
        "heavy_atoms": smiles_tool.heavy_atom_count(mol),
        "mw": molecular_weight(mol),
        "logp": logp(mol),
        "hbd": h_bond_donors(mol),
        "hba": h_bond_acceptors(mol),
        "rings": smiles_tool.ring_count(mol),
        "rotatable_bonds": rotatable_bonds(mol),
        "aromatic_atoms": aromatic_atoms(mol),
        "tpsa": tpsa(mol),
    }
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
