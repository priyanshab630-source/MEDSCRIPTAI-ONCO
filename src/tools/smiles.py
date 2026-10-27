"""A small SMILES reader - just enough to build a molecular graph.

It handles the subset of SMILES that drug-like seed compounds use: atoms,
branches, ring-closure digits, bond symbols and bracket atoms such as [nH] or
[O-]. It is deliberately not a full SMILES engine; for production work swap in
RDKit (`Chem.MolFromSmiles`) behind the same `parse()` signature.
"""

import re

ORGANIC = set("BCNOPSFIbcnops")
TWO_LETTER = {"Cl", "Br"}
AROMATIC = set("bcnops")

ATOMIC_MASS = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.90,
}

BOND_SYMBOL = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5}

_BRACKET_ELEMENT = re.compile(r"^([A-Z][a-z]?|[a-z])")
_BRACKET_H = re.compile(r"H(\d*)")


def _bracket_atom(text):
    """Read element + implicit hydrogens out of a bracket atom like [nH]."""
    inner = text[1:-1]
    match = _BRACKET_ELEMENT.match(inner)
    symbol = match.group(1) if match else "C"
    h_match = _BRACKET_H.search(inner)
    hydrogens = 0
    if h_match:
        hydrogens = int(h_match.group(1)) if h_match.group(1) else 1
    return symbol, hydrogens


def parse(smiles: str) -> dict:
    """Return {"atoms": [...], "bonds": [(i, j, order), ...]}."""
    atoms, bonds = [], []
    stack, ring = [], {}
    prev, pending_bond = None, None
    i = 0

    while i < len(smiles):
        ch = smiles[i]

        if ch == "(":
            stack.append(prev)
            i += 1
            continue

        if ch == ")":
            prev = stack.pop() if stack else prev
            i += 1
            continue

        if ch in BOND_SYMBOL:
            pending_bond = BOND_SYMBOL[ch]
            i += 1
            continue

        if ch == ".":                      # disconnected fragment
            prev, pending_bond = None, None
            i += 1
            continue

        if ch.isdigit():                   # ring closure
            number = int(ch)
            if number in ring:
                partner, order = ring.pop(number)
                bonds.append((partner, prev, order or pending_bond or 1.0))
            else:
                ring[number] = (prev, pending_bond)
            pending_bond = None
            i += 1
            continue

        if ch == "[":                      # bracket atom
            end = smiles.find("]", i)
            end = len(smiles) - 1 if end == -1 else end
            symbol, hydrogens = _bracket_atom(smiles[i:end + 1])
            atoms.append({"symbol": symbol, "aromatic": symbol.islower(),
                          "hydrogens": hydrogens, "bracket": True})
            index = len(atoms) - 1
            if prev is not None:
                bonds.append((prev, index, pending_bond or 1.0))
            prev, pending_bond = index, None
            i = end + 1
            continue

        two = smiles[i:i + 2]
        if two in TWO_LETTER:
            symbol, i = two, i + 2
        elif ch in ORGANIC:
            symbol, i = ch, i + 1
        else:                              # stereochemistry marks, charges, ...
            i += 1
            continue

        atoms.append({"symbol": symbol, "aromatic": symbol in AROMATIC,
                      "hydrogens": 0, "bracket": False})
        index = len(atoms) - 1
        if prev is not None:
            bonds.append((prev, index, pending_bond or 1.0))
        prev, pending_bond = index, None

    return {"atoms": atoms, "bonds": bonds}


def heavy_atom_count(mol: dict) -> int:
    return len(mol["atoms"])


# ---------------------------------------------------------------------------
# Graph helpers: implicit hydrogens, ring bonds, ring count
# ---------------------------------------------------------------------------

VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "F": 1,
           "Cl": 1, "Br": 1, "I": 1, "B": 3}


def adjacency(mol: dict) -> dict:
    """Atom index -> list of (neighbour index, bond order)."""
    adj = {i: [] for i in range(len(mol["atoms"]))}
    for a, b, order in mol["bonds"]:
        adj[a].append((b, order))
        adj[b].append((a, order))
    return adj


def bond_order_sum(mol: dict) -> list:
    adj = adjacency(mol)
    return [round(sum(order for _, order in adj[i]), 2) for i in range(len(mol["atoms"]))]


def implicit_hydrogens(mol: dict) -> list:
    """Valence-filled hydrogens for non-bracket atoms (bracket atoms are explicit)."""
    totals = bond_order_sum(mol)
    result = []
    for i, atom in enumerate(mol["atoms"]):
        if atom["bracket"]:
            result.append(atom["hydrogens"])
            continue
        valence = VALENCE.get(atom["symbol"].capitalize(), 4)
        missing = valence - totals[i]
        result.append(max(0, int(round(missing))))
    return result


def _bridges(mol: dict) -> set:
    """Bond indices that are bridges (i.e. not part of any ring)."""
    n = len(mol["atoms"])
    adj = adjacency(mol)
    edges = [(a, b) for a, b, _ in mol["bonds"]]
    discovery, low, bridges = {}, {}, set()
    timer = [0]

    def walk(node, parent):
        discovery[node] = low[node] = timer[0]
        timer[0] += 1
        for neighbour, _ in adj[node]:
            if neighbour == parent:
                continue
            if neighbour not in discovery:
                walk(neighbour, node)
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    bridges.add(frozenset((node, neighbour)))
            else:
                low[node] = min(low[node], discovery[neighbour])

    for start in range(n):
        if start not in discovery:
            walk(start, -1)
    return bridges


def ring_bond_flags(mol: dict) -> list:
    bridges = _bridges(mol)
    return [frozenset((a, b)) not in bridges for a, b, _ in mol["bonds"]]


def ring_count(mol: dict) -> int:
    """Cyclomatic number: independent rings in the molecule."""
    n = len(mol["atoms"])
    if n == 0:
        return 0
    adj = adjacency(mol)
    seen, components = set(), 0
    for start in range(n):
        if start in seen:
            continue
        components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(neighbour for neighbour, _ in adj[node])
    return max(0, len(mol["bonds"]) - n + components)
