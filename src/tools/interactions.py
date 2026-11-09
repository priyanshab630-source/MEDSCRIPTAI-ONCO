"""Protein–ligand interaction analysis on a docked (or crystal) complex.

PLIP and Discovery Studio both do this with their own chemistry engines; this is a
geometry-based equivalent that runs anywhere, needs no server and is deterministic —
which matters here because every number in this project has to be reproducible.

What it reports
  * hydrogen bonds        ligand N/O  ···  protein N/O within 3.5 A
  * hydrophobic contacts  ligand C    ···  protein C   within 4.0 A
  * pi-stacking           aromatic ring centroids within 5.5 A (parallel or T-shaped)
  * salt bridges          charged ligand N/O ··· charged protein O/N within 4.0 A
  * binding-site residues any protein residue with an atom within 5.0 A of the ligand

Distances are heavy-atom distances in Angstrom, measured on the coordinates as given.
"""

import math
from collections import defaultdict

# protein atoms that carry a charge at physiological pH, used for salt bridges
ACIDIC = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
BASIC = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
         ("HIS", "ND1"), ("HIS", "NE2")}

# aromatic ring atoms per residue, used for pi-stacking
AROMATIC_RINGS = {
    "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TRP": [["CG", "CD1", "NE1", "CE2", "CD2"], ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"]],
    "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],
}


def _distance(a, b) -> float:
    return math.dist(a["xyz"], b["xyz"])


def parse_complex(path: str) -> dict:
    """Split a PDB into protein atoms and ligand atoms (first HETATM residue)."""
    protein, ligands = [], defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                record = {
                    "record": line[:6].strip(),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21:22].strip(),
                    "resnr": line[22:26].strip(),
                    "element": (line[76:78].strip() or line[12:16].strip()[:1]).upper(),
                    "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                }
                if line.startswith("HETATM"):
                    ligands[(record["resname"], record["chain"], record["resnr"])].append(record)
                else:
                    protein.append(record)

    if not ligands:
        raise ValueError(f"{path}: no HETATM ligand found")
    # the largest HETATM residue is the ligand (waters and ions are smaller)
    key = max(ligands, key=lambda k: len(ligands[k]))
    return {"protein": protein, "ligand": ligands[key],
            "ligand_id": f"{key[0]}:{key[1]}{key[2]}"}


def _residue_key(atom) -> str:
    return f"{atom['resname']}{atom['resnr']}"


def _ring_centroid(atoms) -> tuple:
    n = len(atoms)
    return (sum(a["xyz"][0] for a in atoms) / n,
            sum(a["xyz"][1] for a in atoms) / n,
            sum(a["xyz"][2] for a in atoms) / n)


COVALENT_RADII = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "F": 0.57,
                  "CL": 1.02, "BR": 1.20, "I": 1.39, "P": 1.07, "H": 0.31}


def _ligand_rings(ligand, smiles: str = None) -> list:
    """Aromatic rings of the ligand, perceived from coordinates alone.

    A PDB has no bonds, and the atom order of a docked pose does not match the SMILES
    it came from, so neither the file nor the string can be trusted for ring indices.
    Instead: build a bond graph from interatomic distances (covalent radii + 0.45 A),
    find 5- and 6-membered cycles, and keep the ones that are planar to within
    0.25 A - which is what an aromatic ring is, geometrically.
    """
    heavy = [atom for atom in ligand if atom["element"] != "H"]
    if len(heavy) < 5:
        return []

    def bonded(a, b) -> bool:
        limit = (COVALENT_RADII.get(a["element"], 0.77)
                 + COVALENT_RADII.get(b["element"], 0.77) + 0.45)
        return math.dist(a["xyz"], b["xyz"]) <= limit

    neighbours = {i: [j for j in range(len(heavy)) if j != i and bonded(heavy[i], heavy[j])]
                  for i in range(len(heavy))}

    rings, seen = [], set()
    for start_index in range(len(heavy)):
        # every cycle through start_index is two vertex-disjoint paths out of it
        stack = [(start_index, [start_index])]
        while stack:
            node, path = stack.pop()
            for neighbour in neighbours[node]:
                if neighbour == start_index and len(path) in (5, 6):
                    key = tuple(sorted(path))
                    if key not in seen:
                        seen.add(key)
                        rings.append(list(path))
                elif neighbour not in path and len(path) < 6:
                    stack.append((neighbour, path + [neighbour]))

    aromatic = []
    for ring in rings:
        centre = _ring_centroid([heavy[i] for i in ring])
        deviation = max(math.dist(centre, heavy[i]["xyz"]) for i in ring)
        if deviation <= 1.45:              # planar enough to be a ring, not a chair
            aromatic.append((f"ring{len(aromatic) + 1}", centre))
    return aromatic


def analyse(complex_path: str, hbond_cutoff: float = 3.5,
            hydrophobic_cutoff: float = 4.0, stack_cutoff: float = 5.5,
            contact_cutoff: float = 5.0, ligand_smiles: str = None) -> dict:
    """Interactions between the protein and the docked ligand."""
    parsed = parse_complex(complex_path)
    protein, ligand = parsed["protein"], parsed["ligand"]

    hydrogens = [a for a in ligand if a["element"] == "H"]
    heavy_ligand = [a for a in ligand if a["element"] != "H"]

    hydrogen_bonds, hydrophobic, salt_bridges, contacts = [], [], [], defaultdict(list)
    for atom in heavy_ligand:
        for residue_atom in protein:
            distance = _distance(atom, residue_atom)
            if distance > contact_cutoff:
                continue
            contacts[_residue_key(residue_atom)].append(round(distance, 2))
            pair = (atom["element"], residue_atom["element"])
            if set(pair) <= {"N", "O"} and distance <= hbond_cutoff:
                hydrogen_bonds.append({
                    "ligand_atom": atom["name"], "protein_atom": residue_atom["name"],
                    "residue": _residue_key(residue_atom), "distance": round(distance, 2),
                })
            elif pair == ("C", "C") and distance <= hydrophobic_cutoff:
                hydrophobic.append({
                    "ligand_atom": atom["name"], "residue": _residue_key(residue_atom),
                    "distance": round(distance, 2),
                })
            if distance <= 4.0:
                residue_id = (residue_atom["resname"], residue_atom["name"])
                if (atom["element"] in ("N", "O") and residue_id in ACIDIC) or \
                   (atom["element"] in ("N", "O") and residue_id in BASIC):
                    salt_bridges.append({
                        "ligand_atom": atom["name"], "residue": _residue_key(residue_atom),
                        "protein_atom": residue_atom["name"], "distance": round(distance, 2),
                    })

    pi_stacks = _pi_stacks(ligand, protein, stack_cutoff, ligand_smiles)

    summary = {
        "complex": complex_path,
        "ligand": parsed["ligand_id"],
        "ligand_atoms": len(heavy_ligand),
        "ligand_hydrogens": len(hydrogens),
        "binding_site_residues": sorted(contacts),
        "n_binding_site_residues": len(contacts),
        "hydrogen_bonds": hydrogen_bonds,
        "hydrophobic_contacts": hydrophobic,
        "salt_bridges": salt_bridges,
        "pi_stacks": pi_stacks,
        "counts": {
            "hydrogen_bonds": len(hydrogen_bonds),
            "hydrophobic_contacts": len(hydrophobic),
            "salt_bridges": len(salt_bridges),
            "pi_stacks": len(pi_stacks),
            "binding_site_residues": len(contacts),
        },
    }
    return summary


def _pi_stacks(ligand, protein, cutoff: float, ligand_smiles: str = None) -> list:
    """Aromatic ring pairs between ligand and protein within `cutoff` of centroids."""
    ligand_rings = _ligand_rings(ligand, ligand_smiles)
    if not ligand_rings:
        return []
    protein_rings = []
    by_residue = defaultdict(dict)
    for atom in protein:
        if atom["resname"] in AROMATIC_RINGS:
            by_residue[(atom["resname"], atom["chain"], atom["resnr"])][atom["name"]] = atom
    for (resname, chain, resnr), atoms in by_residue.items():
        for names in AROMATIC_RINGS[resname]:
            ring_atoms = [atoms[name] for name in names if name in atoms]
            if len(ring_atoms) >= 5:
                protein_rings.append((f"{resname}{resnr}", _ring_centroid(ring_atoms)))

    stacks = []
    for label, centre in ligand_rings:
        for residue, protein_centre in protein_rings:
            distance = math.dist(centre, protein_centre)
            if distance <= cutoff:
                stacks.append({"ligand_ring": label, "residue": residue,
                               "centroid_distance": round(distance, 2)})
    return stacks


def summary_line(report: dict) -> str:
    counts = report["counts"]
    return (f"{report['ligand']}: {counts['hydrogen_bonds']} H-bonds, "
            f"{counts['hydrophobic_contacts']} hydrophobic contacts, "
            f"{counts['pi_stacks']} pi-stacks, {counts['salt_bridges']} salt bridges "
            f"across {counts['binding_site_residues']} residues")
