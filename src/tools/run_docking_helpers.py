"""Receptor/ligand preparation and the Vina call.

Shared by `scripts/run_docking.py` (the CLI) and `src/tools/docking_run.py` (the
pipeline), so the dashboard and the command line cannot drift apart.
"""

import shutil
import subprocess
from pathlib import Path

ELEMENT_MAP = {"A": "C", "OA": "O", "NA": "N", "SA": "S", "HD": "H", "HS": "H"}


def run_command(command: list, description: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{description} failed:\n{result.stdout}\n{result.stderr}")


def prepare_receptor(pdb_path: Path, out_path: Path) -> Path:
    """PDB -> rigid PDBQT. Open Babel adds Gasteiger charges and AutoDock types."""
    if shutil.which("obabel") is None:
        raise SystemExit("obabel not found - pip install openbabel-wheel")
    raw = out_path.with_suffix(".raw.pdbqt")
    run_command(["obabel", "-ipdb", str(pdb_path), "-opdbqt", "-O", str(raw), "-xr"],
                "receptor preparation")
    # Vina rejects HEADER/COMPND records; keep coordinates and TER only
    kept = [line for line in raw.read_text().splitlines()
            if line.startswith(("ATOM", "HETATM", "TER"))]
    if not kept:
        raise SystemExit(f"{pdb_path}: no receptor atoms after conversion")
    out_path.write_text("\n".join(kept) + "\n")
    raw.unlink(missing_ok=True)
    return out_path


def load_ligand_molecule(path: Path):
    """SDF, MOL2 or SMILES -> RDKit molecule with explicit hydrogens and 3D coords."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        molecule = next(iter(Chem.SDMolSupplier(str(path), removeHs=False)))
    elif suffix in (".mol2", ".ml2"):
        molecule = Chem.MolFromMol2File(str(path), removeHs=False)
    elif suffix in (".smi", ".smiles"):
        molecule = Chem.MolFromSmiles(path.read_text().strip())
    else:
        raise SystemExit(f"unsupported ligand format: {suffix}")
    if molecule is None:
        raise SystemExit(f"{path}: could not parse the ligand")

    if molecule.GetNumConformers() == 0:
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule, randomSeed=0xF00D)
        AllChem.MMFFOptimizeMolecule(molecule)
    elif not any(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()):
        raise SystemExit(f"{path}: ligand has no heavy atoms")
    return molecule


def prepare_ligand(path: Path, out_path: Path):
    """RDKit molecule -> PDBQT with Meeko, plus the SMILES for reporting."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem

    molecule = load_ligand_molecule(path)
    setups = MoleculePreparation().prepare(molecule)
    pdbqt, ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise SystemExit(f"ligand preparation failed: {error}")
    out_path.write_text(pdbqt)
    return Chem.MolToSmiles(Chem.RemoveHs(molecule))


def dock(receptor: Path, ligand: Path, centre, box_size, exhaustiveness: int,
         n_poses: int, out_dir: Path, seed: int = 42) -> list:
    """Run Vina and return the pose energies."""
    import vina

    engine = vina.Vina(sf_name="vina", cpu=0, seed=seed, verbosity=1)
    engine.set_receptor(str(receptor))
    engine.set_ligand_from_file(str(ligand))
    engine.compute_vina_maps(center=list(centre), box_size=[box_size] * 3)
    try:
        # only meaningful when the ligand starts inside the box (redocking);
        # a ligand generated from SMILES has arbitrary coordinates
        crystal_score = float(engine.score()[0])
    except RuntimeError:
        crystal_score = float("nan")
    engine.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    energies = engine.energies(n_poses=n_poses)
    engine.write_poses(str(out_dir / "poses.pdbqt"), n_poses=n_poses, overwrite=True)
    return [[round(float(value), 2) for value in pose] for pose in energies], crystal_score


def build_complex(receptor_pdb: Path, poses: Path, out_path: Path) -> Path:
    """Protein ATOM records + docked pose 1 as HETATM records, in one PDB."""
    text = poses.read_text()
    if "MODEL" not in text:
        raise SystemExit("Vina wrote no poses")
    pose = text.split("MODEL")[1].split("ENDMDL")[0].splitlines()

    protein = [line if line.endswith("\n") else line + "\n"
               for line in receptor_pdb.read_text().splitlines(keepends=True)
               if line.startswith(("ATOM", "TER"))]
    ligand, index = [], 0
    for line in pose:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        index += 1
        element = ELEMENT_MAP.get(line[76:78].strip(), line[76:78].strip() or "C")
        name = f" {element:<3s}"[:4]
        ligand.append(f"HETATM{index:5d} {name} UNL Z   1    {line[30:54]}"
                      f"  1.00  0.00          {element.upper():>2s}\n")

    out_path.write_text("".join(protein) + "".join(ligand) + "END\n")
    return out_path
