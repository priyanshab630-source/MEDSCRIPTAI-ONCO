"""Register a protein structure so the pipeline can dock against it for real.

    # box from the crystal ligand (best: the ligand marks the site you care about)
    python scripts/add_receptor.py --target SHP2 --pdb 3o5x_protein.pdb \
        --reference-ligand 3o5x_ligand.mol2

    # or an explicit centre / residue selection
    python scripts/add_receptor.py --target SHP2 --pdb shp2.pdb --center 31.6 3.9 -1.5
    python scripts/add_receptor.py --target EGFR --pdb egfr.pdb --residues 718,719,745

Writes `data/receptors/<target>.pdb` and `<target>.box.json`, validates that the file
parses as a protein with atoms, and prints the box it chose. From that moment every
hypothesis on that target is docked with AutoDock Vina instead of the surrogate.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                       # noqa: E402
from src.tools import receptors     # noqa: E402

RECEPTOR_DIR = config.DATA_DIR / "receptors"


def read_atoms(path: Path) -> list:
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms.append({
                "name": line[12:16].strip(),
                "resnr": line[22:26].strip(),
                "resname": line[17:20].strip(),
                "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                "hetatm": line.startswith("HETATM"),
            })
    return atoms


def centre_from_ligand(path: Path) -> list:
    suffix = path.suffix.lower()
    if suffix == ".mol2":
        from rdkit import Chem

        molecule = Chem.MolFromMol2File(str(path), removeHs=True)
    elif suffix == ".sdf":
        from rdkit import Chem

        molecule = next(iter(Chem.SDMolSupplier(str(path), removeHs=True)))
    elif suffix in (".pdb", ".pdbqt"):
        atoms = [atom for atom in read_atoms(path) if atom["hetatm"] or suffix == ".pdbqt"]
        if not atoms:
            raise SystemExit(f"{path}: no ligand atoms found")
        return [round(sum(atom["xyz"][i] for atom in atoms) / len(atoms), 2) for i in range(3)]
    else:
        raise SystemExit(f"unsupported reference ligand format: {suffix}")

    if molecule is None:
        raise SystemExit(f"{path}: could not parse the reference ligand")
    positions = molecule.GetConformer().GetPositions()
    return [round(float(sum(column) / len(column)), 2) for column in zip(*positions)]


def centre_from_residues(atoms: list, residues: set) -> list:
    selected = [atom for atom in atoms if atom["resnr"] in residues]
    if not selected:
        raise SystemExit(f"no atoms found for residues {sorted(residues)}")
    return [round(sum(atom["xyz"][i] for atom in selected) / len(selected), 2) for i in range(3)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Register a receptor for docking")
    parser.add_argument("--target", required=True, help="target key, e.g. SHP2")
    parser.add_argument("--pdb", required=True, help="protein structure (PDB)")
    parser.add_argument("--reference-ligand", help="crystal ligand: its centroid becomes the box")
    parser.add_argument("--center", nargs=3, type=float, help="explicit box centre")
    parser.add_argument("--residues", help="comma-separated residue numbers to centre on")
    parser.add_argument("--size", type=float, default=22.0, help="box edge in Angstrom")
    args = parser.parse_args(argv)

    source = Path(args.pdb)
    if not source.is_file():
        raise SystemExit(f"{source} not found")

    atoms = read_atoms(source)
    protein_atoms = [atom for atom in atoms if not atom["hetatm"]]
    if len(protein_atoms) < 100:
        raise SystemExit(f"{source}: only {len(protein_atoms)} protein atoms - "
                         f"this does not look like a receptor")

    if args.center:
        centre = [round(value, 2) for value in args.center]
        how = "explicit --center"
    elif args.reference_ligand:
        centre = centre_from_ligand(Path(args.reference_ligand))
        how = f"centroid of {Path(args.reference_ligand).name}"
    elif args.residues:
        centre = centre_from_residues(atoms, set(args.residues.split(",")))
        how = f"centroid of residues {args.residues}"
    else:
        raise SystemExit("give one of --reference-ligand, --center or --residues")

    RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)
    destination = RECEPTOR_DIR / f"{args.target.lower()}.pdb"
    kept = [line for line in source.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.startswith(("ATOM", "TER"))]
    destination.write_text("\n".join(kept) + "\nEND\n", encoding="utf-8")

    box = {"center": centre, "size": [args.size] * 3,
           "source": f"{how}; structure {source.name}"}
    (RECEPTOR_DIR / f"{args.target.lower()}.box.json").write_text(
        json.dumps(box, indent=2) + "\n", encoding="utf-8")

    print(f"registered {args.target}:")
    print(f"  structure {destination.relative_to(config.ROOT)} "
          f"({len(kept)} lines, {len(protein_atoms)} protein atoms)")
    print(f"  box centre {centre}, {args.size} A cube ({how})")
    print(f"  dockable now: {receptors.available(args.target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
