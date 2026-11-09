# MEDSCRIPTAI-ONCO — how the whole system works

A technical walkthrough of the prototype: what it does, what every agent and file is
for, where the data comes from, where machine learning, deep learning, reinforcement
learning and agentic AI each do their part, and how the dashboard turns all of it into
an answer a scientist can argue with.

---

## 1. What the system is for

Most computational drug-discovery tools answer "which compound looks best?" or "what is
the predicted affinity?" This one answers a different question:

> **Given a claim — drug X modulates target Y in disease Z — is the evidence strong
> enough to say so, and if not, exactly why not?**

It returns one of four verdicts:

| Verdict | Meaning | What triggers it |
|---|---|---|
| `SUPPORTED` | evidence agrees, models are consistent, uncertainty is low | weighted support ≥ 0.30 and overall uncertainty < 0.55 |
| `CONTRADICTED` | the pair-specific literature or the assay argues against the claim | ≥ 50% of literature contradicts and weighted support < 0.35 |
| `UNCERTAIN` | something was found but it does not settle the claim | literature split (≥ 35% contradicting), or uncertainty above threshold |
| `ABSTAIN` | it could not look properly | coverage gap, no assay data, or no graph path, with uncertainty ≥ 0.55 |

`ABSTAIN` is a first-class output, not an error state. Everything else in the system —
the agents, the trained models, the calibration study — exists to make that refusal
trustworthy and to explain which of six specific things went wrong when it happens.

---

## 2. The target: how SHP2 is represented

The architecture is target-agnostic; **SHP2 (PTPN11)** is the case study. A target is
not a structure in this system — it is a *profile* in `data/targets.json`:

```json
"SHP2": {
  "gene": "PTPN11",
  "pocket": { "hydrophobicity": 0.62, "polarity": 0.35,
              "mw_range": [250, 500], "radius_range": [3.2, 6.6],
              "aromatic_rings_range": [1, 3], "reference_mw": 380 },
  "mw": [280, 500], "logp": [1.0, 5.8], "hbd": [0, 3], "hba": [4, 8], "tpsa": [50, 95]
}
```

Two kinds of window, both derived from the target's *known inhibitors* rather than
guessed:

* **descriptor windows** (`mw`, `logp`, `hbd`, `hba`, `tpsa`) — used by the QSAR layer to
  ask "is this compound in the property space where this target's ligands live?"
* **pocket profile** (`hydrophobicity`, `polarity`, `mw_range`, `radius_range`,
  `aromatic_rings_range`) — used by the docking layer to score shape, polarity and
  pharmacophore complementarity.

Three targets ship: `SHP2`, `KRAS_G12C`, `EGFR`. Adding a target is one JSON entry plus,
optionally, literature/KG rows. The drugs themselves live in `data/drugs.json` with
ChEMBL-verified canonical SMILES where available.

---

## 3. One request, end to end

`POST /analyze {"drug": "SHP099", "target": "SHP2", "disease": "KRAS-mutant NSCLC"}`

```
Hypothesis(drug, target, disease, smiles)
        │
        ▼
[1] Hypothesis Planner ──► plan: five checks that would settle the claim
        │
        ▼
[2] Evidence Agent ─────► evidence[]: literature + knowledge-graph paths + assay rows
        │
        ▼
[3] Analysis Agent ─────► computations[]: QSAR, GNN, docking
        │                 tool_calls[]: the orchestration transcript
        │                 model_reliability_measured: MC-dropout spread
        ▼
[4] Critic ─────────────► contradictions[]: reasons the claim could be wrong
        │
        ▼
[5] Adaptive Planner ───► uncertainty vector; loop back to [2] or continue
        │
        ▼
[6] Decision Agent ─────► VerdictReport(verdict, 6 dimensions, evidence, trace)
```

State is one dict (`src/core/state.py`, `OncoState`) that every agent reads and mutates;
each agent appends a line to `trace`, which is why the report can show its own reasoning.

For SHP099 → SHP2 that run produces: `SUPPORTED`, overall uncertainty 0.12, 75% weighted
support, QSAR 0.80, GNN 0.83, docking 0.89 (−10.23 kcal/mol), seven evidence items, and a
tool transcript of `docking → kg_lookup → literature_search`.

---

## 4. The six agents, in detail

Each agent is one file in `src/agents/`, exposes `run(state) -> state`, and does exactly
one job.

### 4.1 Hypothesis Planner (`planner.py`)

**In:** the hypothesis. **Out:** `plan[]`, `loops=0`.

Turns the claim into the checks that would settle it: is there literature on this pair, a
graph path, assay rows, do the models agree, and what would disconfirm it. Nothing is
retrieved here — this agent defines what evidence *would* count, which is what makes the
later abstention criteria meaningful. With an LLM configured it asks the model for the
five checks; otherwise it uses the fixed rule-based plan.

### 4.2 Evidence Agent (`evidence.py`)

**In:** hypothesis, `loops` (to decide whether to widen). **Out:** `evidence[]`.

Three retrievals, each returning `EvidenceItem(source, claim, polarity, weight, meta)`:

* **Literature** (`src/tools/literature.py`) — curated corpus by default, optional live
  Europe PMC. Weight = recency × relevance × specificity. **A stance only counts if the
  paper mentions both the drug and the target**; otherwise it is kept as neutral context
  at 0.35× weight. This rule is the fix for the bug the calibration study found (§11).
* **Knowledge graph** (`src/tools/knowledge_graph.py`) — one- and two-hop paths between
  drug and target, carrying edge polarity. Direct edges weigh 0.55, two-hop paths 0.30.
  Local JSON by default; Neo4j when `NEO4J_URI` is set, same interface.
* **Assays** (`data/assays.json`) — measured IC50/pERK rows, weight 0.70, polarity from
  the measurement.

### 4.3 Computational Analysis Agent (`analysis.py`)

**In:** hypothesis + target profile. **Out:** `computations[]`, `tool_calls[]`,
`model_reliability_measured`.

Runs the three in-silico signals (§6) and then the **tool-orchestration loop** (§7), which
decides in what order the tools are called. It also computes the GNN's Monte-Carlo
dropout spread, which becomes the `model_reliability` dimension.

### 4.4 Critic / Falsification Agent (`critic.py`)

**In:** evidence + computations. **Out:** `contradictions[]`.

Its job is to argue *against* the hypothesis, and it is scored on finding disconfirming
evidence, not on agreement. It reports: negative evidence items, split evidence
(n supporting vs m contradicting), computational disagreement above 0.30 spread, a
missing graph path, missing assay data, and evidence that never mentions the disease.

### 4.5 Adaptive Planner (`adaptive.py`)

**In:** everything so far. **Out:** `uncertainty`, `next_action`, `loops`.

Builds the six-dimension vector (§8) and decides whether another retrieval round is worth
it. If evidence coverage is poor and the loop budget allows, it sets `next_action =
"evidence"` and the graph routes backwards with a widened search; otherwise it routes to
the Decision agent. Capped at `MAX_ADAPTIVE_LOOPS = 2`, so a run always terminates.

### 4.6 Decision Agent (`decision.py`)

**In:** the vector + evidence. **Out:** the `VerdictReport`.

Applies the four-way policy (§8.3) and assembles the report: verdict, six dimensions,
dominant dimension, rationale, support score, evidence, computations, tool transcript,
abstain reasons, and the agent trace.

---

## 5. The evidence layer, and why polarity matters

`EvidenceItem.polarity` is +1 (supports), −1 (contradicts) or 0 (context). The verdict
policy is driven by *weighted* polarity, so both polarity and weight have to be
defensible:

| Source | Weight | Polarity from | Note |
|---|---|---|---|
| Literature (pair-specific) | recency × relevance × 1.0 | study stance | 2024 counts more than 2014 |
| Literature (context only) | recency × relevance × 0.35 | always 0 | a paper on the target is not evidence about the drug |
| Knowledge graph, 1 hop | 0.55 | edge polarity | direct drug→target relation |
| Knowledge graph, 2 hops | 0.30 | any negative hop flips it | e.g. drug→resistance→target |
| Assay | 0.70 | measurement direction | e.g. pERK IC50 of 6.4 µM is negative |

`support_score = Σ(weights of supporting items) / Σ(all weights)`. That single number is
what the calibration study measures for honesty (§11), and what the Brier score is
computed on.

---

## 6. The computational layer (where ML and DL sit)

### 6.1 QSAR — classical ML, `src/tools/qsar.py`

RDKit computes MolWt, MolLogP, HBD, HBA, TPSA, ring count, rotatable bonds and QED. Each
descriptor is compared with the target's window: inside = 1, outside = −0.6; the mean
becomes the QSAR score. A pure-Python graph-estimate fallback keeps the system running
when RDKit is absent, and every report records which backend answered.

### 6.2 Molecular GNN — deep learning, `src/tools/gnn.py` + `gnn_torch.py`

The molecule is a graph: atoms are nodes with element one-hot, degree, aromatic and ring
flags; bonds are edges with order. Two message-passing rounds (`tanh(W_self·x + Â·W_neigh·x)`,
degree-normalised adjacency), mean pooling, dropout 0.15, then a small MLP head. The
**embedding** is what the pipeline uses; the **head** is what makes the training
measurable.

Three checkpoints ship, each trained by `training/train_gnn.py` on a different dataset
(§9):

| Checkpoint | Trained on | Held-out |
|---|---|---|
| `models/gnn.pt` (default) | ESOL, 1,128 solubilities | MAE 0.736, R² 0.754 |
| `models/gnn_affinity.pt` | BACE-1 pIC50, 1,513 IC50s | MAE 0.860, R² 0.423 |
| `models/gnn_chembl_shp2.pt` | ChEMBL SHP2, 759 IC50s | MAE 0.662, R² 0.265 |

The binding score is `sigmoid(1.2 · profile_fit + w·embedding)` — the readout is
rule-based on purpose, and the report says so. The embedding is learned; the affinity
interpretation of it is not.

### 6.3 Docking — physics, `src/tools/docking.py`

RDKit embeds five conformers (ETKDG, seeded), MMFF94 minimises them, and the score
combines size fit, radius fit, aromatic/pharmacophore fit, hydrophobic and polar
complementarity against the pocket profile, times a penalty for floppy ligands. It is
real molecular mechanics on the ligand; the pocket is a profile, not a structure. The
result is mapped to an estimated kcal/mol and is deterministic.

`src/tools/vina_adapter.py` runs **genuine AutoDock Vina** when you supply receptor and
ligand PDBQT files — the library installs and the adapter is tested against its API, but
no protein structure ships.

---

## 7. Where reinforcement learning sits

`src/tools/react_agent.py` + `src/tools/policy.py`.

Originally the five tools were called in a hardcoded order. Now the order is a **learned
policy**:

* **State (11 features):** has-SMILES, budget used, fraction of tools used, whether a
  computational signal exists, whether an evidence signal exists, mean computational
  score, and a one-hot of which tools have been used.
* **Action:** one of `rdkit_descriptors`, `docking`, `gnn_binding`, `kg_lookup`,
  `literature_search`, masked so used tools cannot repeat.
* **Policy network:** MLP 11 → 32 → 5, softmax over the available actions.
* **Reward:** +1.0 for reaching both a computational and an evidence signal, +0.3 × mean
  computational score, −0.1 per tool call, −0.5 if no evidence was gathered.
* **Algorithm:** REINFORCE (policy gradient) with a moving-average baseline (0.9/0.1).
* **Environment:** the real one. Every action executes an actual tool; there is no
  simulator and no cached reward table.
* **Training:** `training/train_policy.py`, 200–300 episodes over the seed drug-target
  pairs; mean return 0.85; checkpoint `models/tool_policy.pt` (4.6 KB).

The learned behaviour is to interleave evidence with compute — `docking → kg_lookup →
literature_search`, three calls to a verdict, where the fixed order spent four calls
before reading any evidence. Three policies coexist and are reported in `tool_mode`:
`rl-policy` (trained checkpoint), `llm` (model chooses), `deterministic` (priority order,
the fallback that keeps runs reproducible without any checkpoint).

---

## 8. The uncertainty model and the verdict policy

### 8.1 Six dimensions (`src/core/uncertainty.py`)

Every component is in [0, 1] where 1 means unresolved:

| Dimension | Weight | Computed from |
|---|---|---|
| `evidence_coverage` | 0.25 | how many of literature / graph / assay were found |
| `assay_consistency` | 0.15 | disagreement among assay rows (+ sample-size penalty) |
| `computational_agreement` | 0.20 | spread between QSAR, GNN and docking scores |
| `graph_support` | 0.15 | agreement of graph paths; 1.0 if no path exists |
| `literature_conflict` | 0.15 | share of pair-specific literature that contradicts |
| `model_reliability` | 0.10 | **MC-dropout spread ÷ training spread**, or a fixed 0.45 penalty when no checkpoint exists |

Overall = weighted mean. The report names the **dominant** dimension, so "uncertain" is
never a shrug — it always points at one of six specific failures.

### 8.2 Why dropout, and not a constant

`gnn_torch.predict_distribution()` runs 25 forward passes with dropout active and returns
mean and standard deviation. The spread is epistemic uncertainty — how much the network
disagrees with itself about a molecule it has not really seen. Measured values: SHP099
0.02, benzene 0.06, ethanol 0.20 under the ESOL encoder. The passes are seeded, so the
same molecule always reports the same number; a framework asking for traceability cannot
have a stochastic verdict.

### 8.3 The policy (`src/core/verdict.py`)

```
abstain criteria triggered AND overall ≥ 0.55   → ABSTAIN
contradicting literature ≥ 50% AND support < 0.35 → CONTRADICTED
contradicting literature ≥ 35%                   → UNCERTAIN
overall ≥ 0.55                                   → UNCERTAIN
support ≥ 0.30                                   → SUPPORTED
otherwise                                        → UNCERTAIN
```

The abstain criteria are explicit and readable — coverage gap ≥ 0.70, no graph path, no
assay data — rather than a threshold buried in a prompt.

---

## 9. Datasets: what ships, where it comes from, and how it is used

Every dataset is registered in `datasets/manifest.json` with source URL, citation,
licence, row count, target column and a SHA-256. `scripts/fetch_datasets.py --verify`
checks every bundled file; the training run refuses a file whose hash does not match.

| Name | Task | Rows | Target | Origin |
|---|---|---|---|---|
| `esol` | regression | 1,128 | measured aqueous solubility | MoleculeNet mirror of Delaney 2004 |
| `bace` | classification | 1,513 | BACE-1 inhibitor class | MoleculeNet / Subramanian 2016 |
| `bace_pic50` | regression | 1,513 | BACE-1 pIC50 (affinity) | same file, pIC50 column |
| `chembl_shp2` | regression | 759 | SHP2 IC50 → pChEMBL | ChEMBL target CHEMBL3864 |
| `qm7` | regression | 7,160 | atomisation energy | MoleculeNet, fetched on demand |

### 9.1 The SHP2 dataset, and the check it earned

`chembl_shp2` was pulled from the ChEMBL API and cleaned by
`scripts/import_chembl_activities.py`: exact `=` relations only, ChEMBL's data-validity
flags respected, one row per compound (repeated measurements collapse to the median
pChEMBL, `n_measurements` kept).

`datasets.ligand_overlap()` then asks a blunt question — *does this table contain
chemistry resembling known inhibitors of the target it claims?* — by Morgan fingerprint:

| Reference | Max similarity | Rows above 0.4 |
|---|---|---|
| TNO155 | 1.00 (exact, pChEMBL 7.96) | 58 |
| RMC-4550 | 0.24 | 0 |
| SHP099 | 0.33 | 0 |

The target identity holds. The check also reveals the *shape* of the set: 93% of compounds
carry ≥14 carbons, median 31 heavy atoms, median pChEMBL 5.42 — an HTS-era table dominated
by one lipophilic acid series, which is why held-out R² is 0.265 rather than 0.75.

### 9.2 How a dataset becomes a model

```
datasets/manifest.json ──► datasets.load(name)  (validate SMILES, drop bad rows)
        │                        │
        │                        ▼
        │              datasets.split()  70/15/15, seeded
        │                        │
        ▼                        ▼
training/train_gnn.py ──► features (atoms, adjacency) ──► GNN ──► head ──► loss
        │                                                              │
        └──► models/<name>.pt + <name>.json  (dataset, citation, sha256,
                                              split sizes, held-out metrics)
```

The metadata file is what the dashboard and `/datasets` read, so the number on screen is
always traceable to the file it was trained on.

---

## 10. Where each discipline is used — the summary table

| Discipline | What it does here | Algorithm / method | Files | Evidence it works |
|---|---|---|---|---|
| **ML (classical)** | Property-space scoring of a compound against the target's inhibitor windows; calibration and evaluation metrics | descriptor windows, weighted evidence scoring, AUROC / Brier / ECE / balanced accuracy | `tools/qsar.py`, `core/uncertainty.py`, `evaluation/calibration.py` | calibration study over 20 labelled cases |
| **DL** | Molecular representation; uncertainty estimation; the verdict writer | 2-layer message-passing GNN (Adam, MSE/BCE); Monte-Carlo dropout; LoRA SFT + DPO | `tools/gnn_torch.py`, `training/train_gnn.py`, `training/sft_train.py`, `training/dpo_train.py` | held-out MAE 0.736 / 0.860 / 0.662 on three datasets |
| **RL** | Which tool to call next, learned from the outcome of the run | REINFORCE with moving-average baseline, masked softmax over 5 actions, 11-feature state | `tools/policy.py`, `training/train_policy.py`, `tools/react_agent.py` | 200 episodes, mean return 0.85, verdict in 3 calls vs 4 |
| **Agentic AI** | The whole pipeline: planning, retrieval, falsification, adaptive retries, and a decision that can abstain | 6-agent graph with one conditional edge; shared typed state; per-agent audit trail; LLM-optional reasoning | `agents/*.py`, `core/graph.py`, `core/engine.py` | four verdicts demonstrated; every run reproducible without an LLM |
| **Physics / chemoinformatics** | Docking and conformer energetics | ETKDG embedding, MMFF94 minimisation, range-based complementarity; Vina adapter | `tools/docking.py`, `tools/vina_adapter.py` | SHP099→SHP2 0.89 vs SHP099→EGFR 0.44 |

---

## 11. The calibration study

`evaluation/` answers the question the design rests on: **does low uncertainty mean a
correct verdict?** Twenty hypotheses — six true pairs with assays, twelve wrong-target
pairs, two unknowns — go through the real pipeline.

| Metric | Value | Reading |
|---|---|---|
| Balanced accuracy | 0.83 | 4/6 true pairs supported, 0/12 wrong pairs supported |
| False-support rate | **0.00** | no wrong pair was ever called `SUPPORTED` |
| Precision / recall | 1.00 / 0.67 | when it says supported it is right; it is conservative |
| Brier score | 0.068 | support score read as P(claim) is well calibrated |
| Expected calibration error | 0.092 | small over-confidence in the 0.4–0.6 band |
| AUROC (uncertainty → correctness) | 1.00 | the vector separates real from wrong pairs |
| Abstention | 100% on unknowns, 5.6% on labelled | abstains when it should |

The study earned its keep immediately: the first run scored 0.28 accuracy and Brier 0.43.
The cause was an evidence bug — any paper mentioning the *target* was counted as support
for *any* drug paired with it, so Osimertinib appeared to inhibit SHP2. Stances now only
count when the paper is about the specific pair, which moved Brier 0.43 → 0.068 and
false-support to zero.

---

## 12. The frontend: how results reach a person

One FastAPI app (`api/app.py`) serves a single-page dashboard (`api/static/index.html`,
no build step, no CDN) plus the JSON endpoints.

| Endpoint | Returns |
|---|---|
| `GET /` | the dashboard |
| `GET /health` | engine, GNN checkpoint, policy, docking backend, tracing sinks, LLM mode |
| `GET /drugs`, `GET /targets` | seed tables, used to populate the form |
| `GET /tools` | tool catalogue + which policy is driving |
| `GET /models` | checkpoint sizes, architectures, algorithms, training metadata |
| `GET /datasets` | provenance for every dataset, the ligand check, and the shipped model's metrics |
| `POST /analyze` | the full six-agent run → `VerdictReport` |
| `POST /uncertainty` | MC-dropout explorer for any SMILES |
| `POST /compare` | several hypotheses in one request |
| `GET /calibration` | the study, cached or recomputed on demand |

The dashboard's nine tabs map one-to-one onto the report, which is the point: nothing is
invented for the UI.

| Tab | Reads | What it shows |
|---|---|---|
| Verdict | `/analyze` | badge, uncertainty gauge, three computational signals, run facts |
| Uncertainty | report.uncertainty | six bars + radar; the model-reliability axis turns green when it is measured, not assumed |
| Tools (RL) | report.tool_calls | the transcript in order, with the policy badge |
| Evidence | report.evidence | every item with stance and weight, filterable by source |
| Agent trace | report.trace | what each of the six agents decided |
| Models & DL | `/models`, `/uncertainty` | checkpoint cards + an MC-dropout explorer: type a SMILES, get predicted value, dropout spread, epistemic uncertainty |
| Datasets | `/datasets` | source, citation, licence, observed range, checksum, ligand check, held-out metrics |
| Calibration | `/calibration` | metric tiles, per-case table, recompute button |
| Compare | `/compare` | several hypotheses side by side |

---

## 13. File map

```
main.py                     CLI: --drug/--target, --demo, --json
config.py                   every setting, env-overridable, all optional
docker-compose.yml          local Neo4j + loader
Dockerfile                  container image for the API
api/app.py                  FastAPI routes (above)
api/static/index.html       the dashboard (vanilla JS, no build)
src/core/
  schemas.py                Hypothesis, EvidenceItem, ComputeResult, UncertaintyVector, VerdictReport
  state.py                  OncoState (shared typed state) + trace logger
  uncertainty.py            the six dimensions and their weights
  verdict.py                four-way policy + abstain criteria
  graph.py                  dependency-free StateGraph with the LangGraph API
  engine.py                 picks real LangGraph or the shim
  tracing.py                local JSONL + optional LangSmith / Langfuse sinks
src/agents/
  planner.py evidence.py analysis.py critic.py adaptive.py decision.py workflow.py
src/tools/
  smiles.py                 SMILES → atom/bond graph, rings, implicit hydrogens
  qsar.py                   RDKit descriptors (or graph estimates) + profile scoring
  gnn.py                    NumPy message passing + binding readout
  gnn_torch.py              trained GNN, MC-dropout uncertainty, checkpoint loading
  docking.py                conformer/MMFF scoring
  vina_adapter.py           real AutoDock Vina
  policy.py                 REINFORCE tool policy
  react_agent.py            tool orchestration loop (rl / llm / deterministic)
  knowledge_graph.py        JSON snapshot + Neo4j backend
  literature.py             curated corpus + optional Europe PMC
  datasets.py               manifest, loading, validation, split, ligand check
  drugs.py                  seed compound table
  llm.py                    Sarvam-compatible client with silent fallback
scripts/
  fetch_datasets.py         --list / --verify / --check / --fetch / --kaggle
  fetch_chembl.py           ChEMBL target → activities → SMILES → manifest
  import_chembl_activities.py  saved ChEMBL export → registered dataset
  load_neo4j.py             load the KG snapshot into Neo4j
  run_local.sh / .ps1       one-command local server
training/
  train_gnn.py              GNN training on any registered dataset (--resume)
  train_policy.py           REINFORCE training for the tool policy
  build_dataset.py          reports → SFT + DPO datasets
  sft_train.py dpo_train.py fine-tuning, with --tiny CPU smoke mode
  tiny_model.py             the local model used by --tiny
evaluation/
  hypotheses.py             20 labelled hypotheses
  calibration.py            AUROC, Brier, ECE, balanced accuracy, abstention
  run_calibration.py        the study, end to end
data/
  targets.json drugs.json assays.json kg_triples.json literature.json
  datasets/                 esol.csv, bace.csv, chembl_shp2.csv
  fixtures/                 recorded arXiv, Europe PMC and ChEMBL responses
datasets/manifest.json      provenance for every dataset
models/                     gnn.pt, gnn_affinity.pt, gnn_chembl_shp2.pt, tool_policy.pt
tests/                      122 tests (5 skipped, each needing optional infrastructure)
```

---

## 14. Testing and CI

122 tests, five skipped (each requires optional infrastructure: a live Neo4j, an
OpenSearch cluster, Docling's models, a Telegram token). Every push runs the suite on
Python 3.11 and 3.12, twice — once on the built-in graph engine and once on real
LangGraph — so the shim cannot drift from the engine. Nothing needs an API key, which is
what makes the numbers reproducible.

---

## 15. What this does not do

* The GNN learns chemistry from solubility and IC50 data; it is not a general affinity
  model, and the affinity readout on top of its embedding is rule-based.
* Docking scores conformers against a pocket *profile*. Real docking needs a structure —
  the Vina adapter is ready for one.
* Neo4j, live literature, LangSmith/Langfuse and the LLM path have code and tests against
  fakes or recorded responses, but have never been exercised against the live services
  from this build environment.
* The calibration set is 20 cases drawn from the same seed corpus, so 0.83 is optimistic
  until an external benchmark runs.
* Fine-tuning executes on a 40k-parameter local model; no real base model has been
  trained.

---

## 16. Running it

```bash
git clone https://github.com/priyanshab630-source/MEDSCRIPTAI-ONCO.git
cd MEDSCRIPTAI-ONCO
bash scripts/run_local.sh          # → http://127.0.0.1:8000

python main.py --demo              # four verdicts in one command
python evaluation/run_calibration.py
python scripts/fetch_datasets.py --verify
```
