# Docking report — UNL:Z1

* receptor: `data/docking/4ajl/4ajl_pocket.pdb`
* ligand: `UNL:Z1` (21 heavy atoms, 3 hydrogens)
* box: 22.0 A cube centred on (18.50, -8.07, 1.93)
* exhaustiveness: 8

## Scores (kcal/mol)

| pose | affinity |
|---|---|
| 1 | -7.19 |
| 2 | -7.13 |
| 3 | -7.12 |
| 4 | -6.51 |
| 5 | -6.38 |

Crystal pose scores -6.76 kcal/mol for comparison.

## Interactions (pose 1)

| type | count |
|---|---|
| hydrogen bonds | 5 |
| hydrophobic contacts | 23 |
| salt bridges | 2 |
| pi-stacks | 0 |
| binding-site residues | 18 |

Binding site: ALA29, ALA95, ALA97, ARG98, ASP51, GLY26, GLY28, GLY31, GLY96, ILE115, ILE119, PHE118, THR94, TYR82, VAL25, VAL27, VAL30, VAL52.

Redocking RMSD: **0.73 A** — reproduced the crystal pose (conventional success threshold 2.0 A).

## Viewers

* `view.pml` — `pymol -cq view.pml`
* `view.cxc` — `chimerax --nogui --script view.cxc`
