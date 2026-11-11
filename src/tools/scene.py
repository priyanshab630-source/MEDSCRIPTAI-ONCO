"""Turn a docked complex into a small JSON scene the browser can draw.

The dashboard has no CDN and no build step, so NGL/3Dmol are out. Instead the server
sends atoms, bonds and interaction lines, and a ~120-line canvas renderer rotates
them. That keeps the page self-contained and makes the geometry inspectable.
"""

import json
import math
from pathlib import Path

COVALENT = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "F": 0.57,
            "CL": 1.02, "BR": 1.20, "I": 1.39, "P": 1.07}


def _distance(a, b) -> float:
    return math.dist(a["xyz"], b["xyz"])


def _read(path: Path) -> tuple:
    protein, ligand = [], []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom = {
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "resnr": line[22:26].strip(),
            "element": (line[76:78].strip() or line[12:16].strip()[:1]).upper(),
            "xyz": [float(line[30:38]), float(line[38:46]), float(line[46:54])],
        }
        (ligand if line.startswith("HETATM") else protein).append(atom)
    return protein, ligand


def build_scene(complex_pdb: str | Path, report: dict, radius: float = 13.0) -> dict:
    """Atoms, bonds and interaction lines within `radius` of the ligand."""
    protein, ligand = _read(Path(complex_pdb))
    if not ligand:
        raise ValueError(f"{complex_pdb}: no ligand atoms")

    centre = [sum(atom["xyz"][i] for atom in ligand) / len(ligand) for i in range(3)]
    near = [atom for atom in protein if math.dist(atom["xyz"], centre) <= radius]

    interacting = set()
    for key in ("hydrogen_bonds", "hydrophobic_contacts", "salt_bridges", "pi_stacks"):
        for entry in report.get(key, []) or []:
            residue = str(entry.get("residue", ""))
            digits = "".join(c for c in residue if c.isdigit())
            if digits:
                interacting.add(digits)

    # protein backbone as a light trace, side chains of interacting residues in full
    atoms = []
    for atom in near:
        atoms.append({
            "x": round(atom["xyz"][0], 2), "y": round(atom["xyz"][1], 2),
            "z": round(atom["xyz"][2], 2), "e": atom["element"],
            "res": f"{atom['resname']}{atom['resnr']}",
            "kind": "interacting" if atom["resnr"] in interacting else "protein",
        })

    heavy = [atom for atom in ligand if atom["element"] != "H"]
    bonds = []
    for i, a in enumerate(heavy):
        for j in range(i + 1, len(heavy)):
            b = heavy[j]
            limit = COVALENT.get(a["element"], 0.77) + COVALENT.get(b["element"], 0.77) + 0.45
            if _distance(a, b) <= limit:
                bonds.append([i, j])

    ligand_payload = [{"x": round(a["xyz"][0], 2), "y": round(a["xyz"][1], 2),
                       "z": round(a["xyz"][2], 2), "e": a["element"],
                       "name": a["name"]} for a in heavy]

    # H-bonds as lines from the ligand atom to the closest atom of the residue
    hbond_lines = []
    for entry in (report.get("plip", {}) or {}).get("hydrogen_bonds", []) or \
                 report.get("hydrogen_bonds", []) or []:
        residue = str(entry.get("residue", ""))
        digits = "".join(c for c in residue if c.isdigit())
        partners = [atom for atom in near if atom["resnr"] == digits]
        if not partners:
            continue
        ligand_atom = entry.get("ligand_atom")
        anchor = next((a for a in heavy if a["name"] == ligand_atom), heavy[0])
        target = min(partners, key=lambda atom: _distance(anchor, atom))
        hbond_lines.append({
            "from": [round(anchor["xyz"][0], 2), round(anchor["xyz"][1], 2),
                     round(anchor["xyz"][2], 2)],
            "to": [round(target["xyz"][0], 2), round(target["xyz"][1], 2),
                   round(target["xyz"][2], 2)],
            "residue": residue,
        })

    # --- backbone trace: consecutive C-alpha atoms, with a break at chain jumps
    ordered = sorted(near, key=lambda atom: (atom["resnr"].zfill(4), atom["name"]))
    backbone, seen_residues, previous = [], set(), None
    for atom in ordered:
        if atom["name"] != "CA" or atom["resnr"] in seen_residues:
            continue
        seen_residues.add(atom["resnr"])
        point = [round(atom["xyz"][0], 2), round(atom["xyz"][1], 2), round(atom["xyz"][2], 2)]
        gap = previous is not None and math.dist(point, previous) > 4.5
        backbone.append({"p": point, "gap": gap,
                         "hot": atom["resnr"] in interacting})
        previous = point

    # --- side-chain sticks for the interacting residues only
    residue_atoms = {}
    for atom in near:
        if atom["resnr"] in interacting:
            residue_atoms.setdefault(atom["resnr"], []).append(atom)
    sticks = []
    for resnr, res_atoms in residue_atoms.items():
        for i, a in enumerate(res_atoms):
            for b in res_atoms[i + 1:]:
                limit = COVALENT.get(a["element"], 0.77) + COVALENT.get(b["element"], 0.77) + 0.45
                if _distance(a, b) <= limit:
                    sticks.append({"a": [round(v, 2) for v in a["xyz"]],
                                   "b": [round(v, 2) for v in b["xyz"]],
                                   "res": f"{a['resname']}{a['resnr']}"})

    labels = []
    seen = set()
    for atom in near:
        label = f"{atom['resname']}{atom['resnr']}"
        if atom["resnr"] in interacting and label not in seen and atom["name"] == "CA":
            seen.add(label)
            labels.append({"text": label, "x": round(atom["xyz"][0], 2),
                           "y": round(atom["xyz"][1], 2), "z": round(atom["xyz"][2], 2)})

    return {
        "centre": [round(value, 2) for value in centre],
        "radius": radius,
        "protein": atoms,
        "ligand": ligand_payload,
        "bonds": bonds,
        "hbonds": hbond_lines,
        "labels": labels,
        "backbone": backbone,
        "sticks": sticks,
        "counts": {"protein_atoms": len(atoms), "ligand_atoms": len(ligand_payload),
                   "bonds": len(bonds), "hbonds": len(hbond_lines),
                   "backbone_points": len(backbone), "residue_sticks": len(sticks)},
    }


def write_scene(complex_pdb: str | Path, report: dict, out_path: str | Path) -> Path:
    scene = build_scene(complex_pdb, report)
    Path(out_path).write_text(json.dumps(scene))
    return Path(out_path)
