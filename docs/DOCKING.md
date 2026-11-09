# Docking, visualisation and interaction analysis

Steps 3–5 of a structure-based workflow — dock, visualise, analyse — run end to end
from one command, on real coordinates.

```bash
pip install vina meeko openbabel-wheel rdkit

python scripts/run_docking.py \
    --receptor data/docking/4ajl/4ajl_pocket.pdb \
    --ligand   data/docking/4ajl/4ajl_ligand.mol2 \
    --out runs/4ajl --exhaustiveness 8 \
    --crystal-ligand data/docking/4ajl/4ajl_ligand.mol2
```

## What the script does

| Step | Tool | Output |
|---|---|---|
| 1. Prepare receptor | Open Babel (`-xr`, rigid) | `receptor.pdbqt` |
| 2. Prepare ligand | Meeko (hydrogens, AutoDock types, rotatable bonds) | `ligand.pdbqt` |
| 3. Dock | **AutoDock Vina 1.2** (box centred on the ligand or `--center`) | `poses.pdbqt`, energies |
| 4. Build complex | protein ATOM + best pose HETATM | `complex.pdb` |
| 5. Analyse interactions | `src/tools/interactions.py` | `report.json`, `report.md` |
| 6. Validate (optional) | RDKit `GetBestRMS` vs crystal ligand | redocking RMSD |
| 7. Viewer scripts | template + measured numbers | `view.pml`, `view.cxc` |

## The validation that matters

The bundled fixture is PDBbind complex **4AJL** (a real protein–ligand crystal
structure with measured affinity). Docking the crystal ligand back into its own pocket
reproduces the experimental pose:

| Measure | Result |
|---|---|
| Redocking RMSD, pose 1 vs crystal | **0.73 Å** (success threshold is conventionally 2.0 Å) |
| Best pose affinity | −7.19 kcal/mol |
| Crystal pose scored by Vina | −6.76 kcal/mol |
| Hydrogen bonds | 5 |
| Hydrophobic contacts | 23 |
| Salt bridges | 2 |
| π-stacks | 0 — nearest protein aromatic ring is 7.8 Å away, so none exist in this pose |
| Binding-site residues | 18 |

That is the check to run before trusting any docking number: if redocking fails, the
box, the protonation or the receptor preparation is wrong.

## Interaction analysis

`src/tools/interactions.py` is geometry-based and deterministic — no server, no GUI:

* **hydrogen bonds** — ligand N/O ··· protein N/O within 3.5 Å
* **hydrophobic contacts** — ligand C ··· protein C within 4.0 Å
* **salt bridges** — charged ligand N/O ··· ASP/GLU/LYS/ARG/HIS atoms within 4.0 Å
* **π-stacking** — aromatic ring centroids within 5.5 Å; ligand rings are perceived
  from coordinates alone (distance-graph cycles + planarity), because a PDB carries no
  bond orders and the pose atom order does not match the SMILES it came from
* **binding-site residues** — any residue with an atom within 5.0 Å of the ligand

PLIP is also wired as an option (`plip -f complex.pdb -x -o report`), but on this
complex PLIP returned no interactions despite identifying the ligand correctly, which
is why the built-in analyser is the default: it always produces a number you can
trace to a distance.

## Visualisation

PyMOL and ChimeraX are not installed in the build environment, so the script **writes
their session scripts** rather than screenshots. Both are ready to run on your machine:

```bash
pymol -cq runs/4ajl/view.pml        # → docking.png
chimerax --nogui --script runs/4ajl/view.cxc   # → docking.png
```

Each script loads `complex.pdb`, cartoons the protein, sticks the ligand and the
residues within 4.5 Å, draws hydrogen bonds, labels the binding site, and renders a
1600×1200 PNG. Discovery Studio is a licensed desktop product with no scriptable
headless equivalent here — the JSON report carries the same interaction table it would
show, so you can reproduce the analysis there by hand.

## Using your own structure

```bash
# 1. a receptor: any PDB, ideally the pocket or the whole chain
python scripts/run_docking.py --receptor my_protein.pdb --ligand my_ligand.sdf \
    --out runs/mine --center 12.4 -3.1 28.9 --box-size 24

# 2. a ligand from SMILES (coordinates are generated with ETKDG + MMFF)
echo 'CC(=O)Oc1ccccc1C(=O)O' > aspirin.smi
python scripts/run_docking.py --receptor my_protein.pdb --ligand aspirin.smi --out runs/aspirin
```

For SHP2 specifically: the repo ships the target *profile* but no structure. Supply any
SHP2 PDB (allosteric tunnel or catalytic domain), pick the box centre from a bound
ligand or the pocket residues, and the same command produces a real docked pose with
interaction analysis.
