"""Run molecular docking end to end: prepare, dock, analyse, write viewer scripts.

    pip install vina meeko openbabel-wheel rdkit

    python scripts/run_docking.py \
        --receptor data/docking/4ajl/4ajl_pocket.pdb \
        --ligand   data/docking/4ajl/4ajl_ligand.mol2 \
        --out runs/4ajl --exhaustiveness 8

What it does, in order:

  1. **prepare the receptor** - PDB -> PDBQT (rigid) with Open Babel
  2. **prepare the ligand**   - SDF / MOL2 / SMILES -> PDBQT with Meeko (adds hydrogens,
     assigns AutoDock types and rotatable bonds)
  3. **dock**                 - AutoDock Vina, box centred on the ligand (or `--center`)
  4. **build the complex**    - protein + best pose as one PDB, ready for a viewer
  5. **analyse interactions** - H-bonds, hydrophobic contacts, salt bridges, pi-stacks,
     binding-site residues (`src/tools/interactions.py`)
  6. **validate, if asked**   - redocking RMSD of pose 1 against the crystal ligand
  7. **write viewers**        - PyMOL `.pml` and ChimeraX `.cxc` scripts, plus a JSON and
     a Markdown report

Every number in the report comes from the files in `--out`; nothing is estimated.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools import interactions, plip_adapter  # noqa: E402
from src.tools.run_docking_helpers import (  # noqa: E402
    build_complex, dock, load_ligand_molecule, prepare_ligand, prepare_receptor,
    run_command)



def redocking_rmsd(complex_pdb: Path, crystal_ligand: Path) -> float:
    """Symmetry-corrected RMSD between pose 1 and the crystal ligand, or NaN."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolAlign

    RDLogger.DisableLog("rdApp.*")
    reference = load_ligand_molecule(crystal_ligand)
    reference = Chem.RemoveHs(reference)

    # pose 1 as PDB through Open Babel so RDKit can perceive bonds
    pose_pdb = complex_pdb.with_name("pose1.pdb")
    run_command(["obabel", str(complex_pdb.with_name("poses.pdbqt")),
                 "-opdb", "-O", str(pose_pdb), "-f", "1", "-l", "1"],
                "pose extraction")
    probe = Chem.MolFromPDBFile(str(pose_pdb), removeHs=True, sanitize=True)
    if probe is None or probe.GetNumAtoms() != reference.GetNumAtoms():
        return float("nan")
    try:
        return round(rdMolAlign.GetBestRMS(probe, reference), 2)
    except Exception:                       # noqa: BLE001 - graph mismatch is not fatal
        return float("nan")


def pymol_script(complex_pdb: Path, report: dict, out_path: Path) -> Path:
    residues = " or ".join(f"resi {r.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
                           for r in report["binding_site_residues"][:12]) or "polymer"
    out_path.write_text(f"""# PyMOL session for {complex_pdb.name}
#   pymol -cq {out_path.name}          (headless, writes docking.png)
load {complex_pdb.resolve()}, complex
hide everything
show cartoon, polymer
color grey70, polymer
select ligand, resn UNL
show sticks, ligand
color yellow, ligand
select pocket, byres (polymer within 4.5 of ligand)
show sticks, pocket and not name N+C+O
color cyan, pocket and elem C
distance hbonds, ligand, polymer, 3.5, mode=2
show dashes, hbonds
color red, hbonds
label pocket and name CA, "%s%s" % (resn, resi)
set label_size, 14
orient ligand
bg_color white
ray 1600, 1200
png docking.png, dpi=200
""")
    return out_path


def chimerax_script(complex_pdb: Path, report: dict, out_path: Path) -> Path:
    out_path.write_text(f"""# ChimeraX session for {complex_pdb.name}
#   chimerax --nogui --script {out_path.name}
open {complex_pdb.resolve()}
hide atoms
show cartoon
color #1 grey
select #1:UNL
show sel atoms
style sel stick
color sel yellow
select zone #1:UNL 4.5
show sel atoms
style sel stick
color sel cyan
hbonds #1 & protein restrict #1:UNL reveal true color red
label #1 & protein & @CA residues text "{{resname}}{{resseq}}" color black
view #1:UNL
save docking.png width 1600 height 1200 supersample 3
""")
    return out_path


def markdown_report(report: dict, out_path: Path) -> Path:
    counts = report["counts"]
    lines = [
        f"# Docking report — {report['ligand']}",
        "",
        f"* receptor: `{report['receptor']}`",
        f"* ligand: `{report['ligand']}` ({report['ligand_atoms']} heavy atoms, "
        f"{report['ligand_hydrogens']} hydrogens)",
        f"* box: {report['box_size']} A cube centred on "
        f"({report['box_centre'][0]:.2f}, {report['box_centre'][1]:.2f}, "
        f"{report['box_centre'][2]:.2f})",
        f"* exhaustiveness: {report['exhaustiveness']}",
        "",
        "## Scores (kcal/mol)",
        "",
        "| pose | affinity |",
        "|---|---|",
    ]
    for index, energy in enumerate(report["poses"][:5], start=1):
        lines.append(f"| {index} | {energy[0]} |")
    lines += [
        "",
        f"Crystal pose scores {report['crystal_pose_score']} kcal/mol for comparison.",
        "",
        "## Interactions (pose 1)",
        "",
        "| type | count |",
        "|---|---|",
        f"| hydrogen bonds | {counts['hydrogen_bonds']} |",
        f"| hydrophobic contacts | {counts['hydrophobic_contacts']} |",
        f"| salt bridges | {counts['salt_bridges']} |",
        f"| pi-stacks | {counts['pi_stacks']} |",
        f"| binding-site residues | {counts['binding_site_residues']} |",
        "",
        "Binding site: " + ", ".join(report["binding_site_residues"]) + ".",
        "",
    ]
    if report.get("redocking_rmsd") is not None:
        value = report["redocking_rmsd"]
        verdict = ("reproduced the crystal pose" if value == value and value <= 2.0
                   else "did not reproduce the crystal pose")
        lines += [f"Redocking RMSD: **{value} A** — {verdict} "
                  f"(conventional success threshold 2.0 A).", ""]
    lines += ["## Viewers", "",
              "* `view.pml` — `pymol -cq view.pml`",
              "* `view.cxc` — `chimerax --nogui --script view.cxc`", ""]
    out_path.write_text("\n".join(lines))
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dock a ligand and analyse the pose")
    parser.add_argument("--receptor", required=True, help="protein PDB (pocket is fine)")
    parser.add_argument("--ligand", required=True, help="ligand SDF, MOL2 or SMILES")
    parser.add_argument("--out", default="runs/docking")
    parser.add_argument("--center", nargs=3, type=float, default=None,
                        help="box centre; defaults to the ligand's centroid")
    parser.add_argument("--box-size", type=float, default=22.0)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--n-poses", type=int, default=9)
    parser.add_argument("--crystal-ligand", help="crystal ligand, to compute redocking RMSD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plip", dest="plip", action="store_true", default=None,
                        help="also profile interactions with PLIP")
    parser.add_argument("--no-plip", dest="plip", action="store_false")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdb = Path(args.receptor)
    ligand_path = Path(args.ligand)

    print(f"1/6 preparing receptor  {receptor_pdb.name}")
    receptor_pdbqt = prepare_receptor(receptor_pdb, out_dir / "receptor.pdbqt")

    print(f"2/6 preparing ligand    {ligand_path.name}")
    smiles = prepare_ligand(ligand_path, out_dir / "ligand.pdbqt")

    molecule = load_ligand_molecule(ligand_path)
    if args.center:
        centre = tuple(args.center)
    else:
        positions = molecule.GetConformer().GetPositions()
        centre = tuple(float(value) for value in positions.mean(axis=0))
    print(f"    box centre ({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}), "
          f"{args.box_size} A cube")

    print("3/6 docking with AutoDock Vina")
    poses, crystal_score = dock(receptor_pdbqt, out_dir / "ligand.pdbqt", centre,
                                args.box_size, args.exhaustiveness, args.n_poses,
                                out_dir, args.seed)
    print(f"    best pose {poses[0][0]} kcal/mol (crystal pose {crystal_score:.2f})")

    print("4/6 building the complex")
    complex_pdb = build_complex(receptor_pdb, out_dir / "poses.pdbqt",
                                out_dir / "complex.pdb")

    print("5/6 analysing interactions")
    report = interactions.analyse(str(complex_pdb), ligand_smiles=smiles)
    report.update({
        "receptor": str(receptor_pdb),
        "ligand_file": str(ligand_path),
        "ligand_smiles": smiles,
        "poses": poses,
        "crystal_pose_score": round(crystal_score, 2),
        "box_centre": [round(value, 3) for value in centre],
        "box_size": args.box_size,
        "exhaustiveness": args.exhaustiveness,
        "engine": "AutoDock Vina 1.2",
        "redocking_rmsd": None,
    })
    if args.crystal_ligand:
        rmsd = redocking_rmsd(complex_pdb, Path(args.crystal_ligand))
        report["redocking_rmsd"] = None if rmsd != rmsd else rmsd

    use_plip = plip_adapter.available() if args.plip is None else args.plip
    if use_plip:
        print("    profiling with PLIP")
        report["plip"] = plip_adapter.analyse(complex_pdb, out_dir / "plip")
        print("   ", plip_adapter.summary_line(report["plip"]))

    print("6/6 writing reports and viewer scripts")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown_report(report, out_dir / "report.md")
    pymol_script(complex_pdb, report, out_dir / "view.pml")
    chimerax_script(complex_pdb, report, out_dir / "view.cxc")

    print("\n" + interactions.summary_line(report))
    if report.get("plip"):
        print(plip_adapter.summary_line(report["plip"]))
    print(f"binding site: {', '.join(report['binding_site_residues'][:12])}")
    if report["redocking_rmsd"] is not None:
        print(f"redocking RMSD: {report['redocking_rmsd']} A")
    print(f"\nwritten to {out_dir}/: report.json, report.md, complex.pdb, "
          f"poses.pdbqt, view.pml, view.cxc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
