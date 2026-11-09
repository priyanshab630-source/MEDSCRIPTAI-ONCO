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
| `chembl_shp2` | regression | 759 | SHP2 (PTPN11) IC50 → pChEMBL | ChEMBL, target CHEMBL3864 |

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

## ChEMBL (real bioactivity, fetched by you)

`scripts/fetch_chembl.py` talks to the ChEMBL REST API directly — target resolution,
paged activity retrieval, canonical SMILES joining, then a CSV plus a manifest entry:

```bash
python scripts/fetch_chembl.py --target PTPN11 --name chembl_shp2      # SHP2
python scripts/fetch_chembl.py --target EGFR --type IC50 --limit 2000
python scripts/fetch_chembl.py --chembl-id CHEMBL3351120 --name chembl_x
python training/train_gnn.py --dataset chembl_shp2 --epochs 40
```

It records the target ChEMBL id, assay type, retrieval date, row count and SHA-256,
and notes the licence (**ChEMBL is CC BY-SA 4.0** — attribute and share alike).

This build sandbox cannot reach `www.ebi.ac.uk`, so the fetcher is tested against
recorded responses in `data/fixtures/chembl_*.json` rather than a live call. On a
machine with normal network access the same command performs the real download.

### Importing activities you already hold

If you have ChEMBL activity records already (pulled through the ChEMBL MCP
connector, or saved from `/activity.json`), skip the network step:

```bash
python scripts/import_chembl_activities.py activities.json \
    --target-id CHEMBL3864 --target-name "SHP2 (PTPN11)" --name chembl_shp2
```

It keeps exact (`=`) IC50 rows that have a pChEMBL value and no ChEMBL validity
flag, collapses repeated measurements to the median per compound, and registers
the CSV like any other dataset.

**`chembl_shp2`** (bundled): 759 unique SHP2 compounds, from the first 1,000 of
3,404 pChEMBL-bearing IC50 records for `CHEMBL3864`, retrieved 2026-11-09 from
ChEMBL v34. This is a partial slice, not the whole target, and it is heavy in
phosphatase-assay hits (natural products, phosphopeptides), not just
allosteric-site inhibitors. Re-run with a larger export before drawing
conclusions from a model trained on it.

## SHP2 bioactivity from ChEMBL (`chembl_shp2`)

759 compounds with IC50 measurements against SHP2 (PTPN11), target
`CHEMBL3864`, fetched from the ChEMBL API and cleaned by
`scripts/import_chembl_activities.py`: exact `=` relations only, ChEMBL's own
data-validity flags respected, one row per compound (repeated measurements
collapse to the median pChEMBL, with `n_measurements` kept).

```bash
python scripts/fetch_datasets.py --check chembl_shp2     # is it the right target?
python training/train_gnn.py --dataset chembl_shp2 --epochs 40 --out models/gnn_chembl_shp2.pt
```

**`--check` is the guard this dataset earned.** It measures the closest analogue
in the file to each known inhibitor of the claimed target, by Morgan fingerprint:

| Reference | Max similarity | Rows above 0.4 |
|---|---|---|
| TNO155 | 1.00 | 58 |
| RMC-4550 | 0.24 | 0 |
| SHP099 | 0.33 | 0 |

TNO155 is present at pChEMBL 7.96, and 58 compounds sit within 0.4 of it — so the
target identity holds. What the check also shows is the *shape* of the set: 93% of
the compounds carry 14 or more carbons, the median is 31 heavy atoms, and only a
minority are potent. It is an HTS-era SHP2 table dominated by one lipophilic acid
series, not a curated lead-optimisation set — which is why the median pChEMBL is
5.42 and why the model's R² on held-out data is 0.265 rather than 0.75.

## Affinity data that also ships: BACE-1 pIC50

`bace_pic50` is a regression entry over the same bundled BACE file: **1,513 measured
IC50 values** converted to pChEMBL (pIC50 = −log₁₀ IC50, range 2.54–10.52). That is
real binding affinity, and it is what the affinity checkpoint below is trained on.

```bash
python training/train_gnn.py --dataset bace_pic50 --epochs 40 --out models/gnn_affinity.pt
```

## Two checkpoints

| Checkpoint | Trained on | Held-out | Use |
|---|---|---|---|
| `models/gnn.pt` (default) | ESOL, 1,128 solubilities | MAE 0.736, R² 0.754 | broad, drug-like chemical encoder |
| `models/gnn_affinity.pt` | BACE-1 pIC50, 1,513 IC50s | MAE 0.860, RMSE 1.054, R² 0.423 | affinity-trained encoder |
| `models/gnn_chembl_shp2.pt` | ChEMBL SHP2, 759 IC50s | MAE 0.662, RMSE 0.923, R² 0.265 | affinity encoder for the project's own case-study target |

```bash
GNN_CHECKPOINT=models/gnn_affinity.pt uvicorn api.app:app --port 8000
```

The affinity encoder is specialised to BACE-1, so it reports high epistemic
uncertainty on unrelated chemistry (a SHP2 inhibitor scores 0.85 on MC dropout,
versus 0.02 under the broader ESOL encoder) — which is the uncertainty model
behaving correctly, and the reason the ESOL encoder stays the default.

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
