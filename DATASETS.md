# Datasets and where they come from

Every dataset this project trains on is registered in
[`datasets/manifest.json`](datasets/manifest.json) with its source URL, citation,
licence and SHA-256. Nothing is downloaded silently, and nothing is trained on a
file whose hash does not match.

```bash
python scripts/fetch_datasets.py --list      # what is registered, and where from
python scripts/fetch_datasets.py --verify    # checksum every bundled file
```

## Bundled (train offline, reproducible)

| Name | Task | Rows | Target | Source |
|---|---|---|---|---|
| `esol` | regression | 1,128 | measured aqueous solubility (log10 mol/L) | MoleculeNet / Delaney 2004 |
| `bace` | classification | 1,513 | BACE-1 inhibitor class (0/1) | MoleculeNet / Subramanian 2016 |

**ESOL — Delaney aqueous solubility.** 1,128 drug-like molecules with measured
solubility. Downloaded from the MoleculeNet mirror
(`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv`).
Cite: *Delaney, J. S. (2004) ESOL: Estimating Aqueous Solubility Directly from
Molecular Structure. J. Chem. Inf. Comput. Sci. 44(3):1000–1005*, packaged by
*Wu et al. (2018), MoleculeNet, Chem. Sci. 9:513–530*.

**BACE — BACE-1 inhibition.** 1,513 compounds with binary inhibition labels, from
the same MoleculeNet mirror. The original file carries 588 fingerprint columns
(3.9 MB); the version in this repo keeps only `smiles`, `label` and `pIC50`
(116 KB) because the GNN learns its own features from the graph. Cite:
*Subramanian et al. (2016)* and *Wu et al. (2018)*.

Both files are research data redistributed by the MoleculeNet authors. Check the
upstream terms before republishing them elsewhere.

## Fetchable on demand

| Name | Task | Rows | Note |
|---|---|---|---|
| `qm7` | regression | 7,160 | atomisation energies; not committed because of size |

```bash
python scripts/fetch_datasets.py --fetch qm7
```

## Kaggle

Kaggle needs your own API token, and the environment this repo was built in could
not reach `kaggle.com`, so **no Kaggle data is bundled** — the path is wired and
documented rather than pretended.

```bash
pip install kaggle
# kaggle.com -> Settings -> API -> Create New Token -> save to ~/.kaggle/kaggle.json

python scripts/fetch_datasets.py --kaggle-search "chembl bioactivity"
python scripts/fetch_datasets.py --kaggle <owner>/<dataset-slug> \
    --name chembl_bioactivity --smiles-col smiles --target-col pIC50 \
    --task regression
```

The fetch records the slug, the dataset URL, the row count and the checksum into
`datasets/manifest.json`, so provenance survives even for data you bring yourself.
Two cautions: each Kaggle dataset carries its own licence (do not redistribute
without checking), and column names vary — pass `--smiles-col` / `--target-col`
to match the file you downloaded.

## What the data is used for

```bash
python training/train_gnn.py --dataset esol --epochs 30      # regression
python training/train_gnn.py --dataset bace --epochs 30      # classification
```

The graph encoder is trained on the dataset's measured values; the checkpoint and
its metadata (`models/gnn.json`) record the dataset name, citation, file hash,
split sizes and held-out metrics.

Result of the committed checkpoint (`models/gnn.pt`, ESOL, 30 epochs, seed 7):

| Split | Molecules | MAE | RMSE | R² |
|---|---|---|---|---|
| train | 790 | — | — | — |
| validation | 169 | 0.899 | 1.187 | 0.663 |
| **test** | **169** | **0.736** | **0.969** | **0.754** |

For context, Delaney's original ESOL linear model reports RMSE ≈ 0.75–1.0 log
units on this set, so the GNN is in the expected range — not a breakthrough, and
not a claim about binding affinity. It is a real model on real measurements, which
is what the proxy-task version was not.

## Adding your own dataset

1. Put the CSV somewhere under `data/datasets/`.
2. Add an entry to `datasets/manifest.json` with `name`, `file`, `task`,
   `smiles_column`, `target_column`, `source`, `url`, `citation`, `license`.
3. Run `python scripts/fetch_datasets.py --verify` to record the checksum.
4. `python training/train_gnn.py --dataset <name>`.
