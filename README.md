# MEDSCRIPTAI-ONCO

**Evidence-grounded, contradiction-aware agentic framework for validating drug–target hypotheses.**

Most AI-for-drug-discovery demos rank or generate compounds. This one does the
opposite: you hand it a claim — *"SHP099 modulates SHP2 in KRAS-mutant NSCLC"* — and
it tries to **falsify** it using molecular/bioactivity computation, a knowledge
graph and literature evidence, then returns one of four honest outcomes:

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | evidence agrees, models are consistent, uncertainty is low |
| `CONTRADICTED` | the literature or assays argue against the claim |
| `UNCERTAIN` | evidence is split or thin — the claim is not settled either way |
| `ABSTAIN` | the framework refuses to answer: it could not look properly |

The architecture is target-agnostic. **SHP2 (PTPN11) is the first case study.**

---

## Why it is built this way

1. **Falsification first.** A dedicated Critic agent is rewarded for finding
   disconfirming evidence, not for agreeing. Confirmation bias is a design bug,
   so the loop has an agent whose only job is to break the hypothesis.
2. **Six uncertainty dimensions, not one confidence number.** A single 0.82 hides
   *why* the system is unsure. The vector exposes whether the doubt is missing
   evidence, conflicting assays, disagreeing models, a weak graph path,
   contradictory literature, or the fact that the models are untrained.
3. **Explicit abstention.** `src/core/verdict.py` contains the actual criteria
   (evidence coverage, missing assay data, no graph path) — you can read and
   challenge them, rather than trusting a threshold buried in a prompt.
4. **Traceability over generation.** Every claim in the report is attached to a
   source record and every agent step is logged in `report["trace"]`.

---

## Architecture

```
                      ┌──────────────────────────┐
   hypothesis  ─────► │ 1. Hypothesis Planner    │  what would settle this claim?
                      └───────────┬──────────────┘
                                  ▼
                      ┌──────────────────────────┐   literature (local corpus
                      │ 2. Evidence Agent        │   + optional Europe PMC),
                      │                          │   knowledge graph (JSON or
                      └───────────┬──────────────┘   Neo4j), assay table
                                  ▼
                      ┌──────────────────────────┐
                      │ 3. Computational Analysis│   QSAR · molecular GNN ·
                      │                          │   surrogate docking
                      └───────────┬──────────────┘
                                  ▼
                      ┌──────────────────────────┐
                      │ 4. Falsification / Critic│   argues against the claim
                      └───────────┬──────────────┘
                                  ▼
                      ┌──────────────────────────┐   coverage gap + loops left?
                      │ 5. Adaptive Planner      │──── yes ──► back to (2),
                      └───────────┬──────────────┘   with a wider search
                                  │ no
                                  ▼
                      ┌──────────────────────────┐
                      │ 6. Decision Agent        │  6-dim vector ──► verdict
                      └───────────┬──────────────┘
                                  ▼
                        SUPPORTED / CONTRADICTED /
                        UNCERTAIN / ABSTAIN + evidence trail
```

The graph is a `StateGraph` with one conditional edge (the adaptive loop). It
runs on the **real LangGraph engine** when `USE_LANGGRAPH=1`, and on a
dependency-free shim with the same API otherwise — same nodes, same edges, same
verdict. `tests/test_engine.py` runs the suite on both and asserts they agree.

---

## Quickstart

```bash
git clone https://github.com/priyanshab630-source/MEDSCRIPTAI-ONCO.git
cd MEDSCRIPTAI-ONCO

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --demo                                  # five cases, no keys needed
python main.py --drug SHP099 --target SHP2 --disease "KRAS-mutant NSCLC"
python main.py --drug Adagrasib --target KRAS_G12C --json
```

Web UI:

```bash
uvicorn api.app:app --reload --port 8000
# open http://localhost:8000
```

### The demo cases

| # | Hypothesis | Expected |
|---|---|---|
| 1 | SHP099 → SHP2 in KRAS-mutant NSCLC | `SUPPORTED` |
| 2 | SHP099 → SHP2 in KRAS-mutant CRC | `UNCERTAIN` (literature is split) |
| 3 | RMC-4550 → SHP2 in KRAS-mutant CRC | `CONTRADICTED` |
| 4 | Adagrasib → KRAS_G12C in NSCLC | `SUPPORTED` |
| 5 | CompoundX → SHP2 | `ABSTAIN` |

### Optional API keys

Everything works offline. Add keys only if you want LLM-written reasoning and
live literature:

```bash
cp .env.example .env
# SARVAM_API_KEY=...        # Sarvam AI, OpenAI-compatible
# NEO4J_URI=bolt://...      # real knowledge graph instead of the JSON snapshot
```

With no key, agents fall back to deterministic rule-based reasoning and the run
stays reproducible. `GET /health` tells you which mode is active.

### Real knowledge graph (Neo4j)

The JSON snapshot is the default so the prototype runs with no database. To use
a real graph:

```bash
docker compose up -d neo4j          # Neo4j 5 + browser on :7474
docker compose run --rm loader      # loads data/kg_triples.json
NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=testpassword \
    python main.py --drug SHP099 --target SHP2
```

Edges are stored as `(:Entity)-[:RELATED {relation, polarity, source}]->(:Entity)`
so new relation types need no migration. The Cypher is exercised in
`tests/test_neo4j.py` against a fake driver, so the query shape is tested even
where no database is available.

### Observability

Every run writes one JSON line to `traces/runs.jsonl` (agent steps, inputs,
outputs, duration). LangSmith and Langfuse are opt-in through environment
variables and are best-effort — an observability outage never breaks a run:

```bash
LANGSMITH_API_KEY=... LANGSMITH_PROJECT=medscriptai-onco python main.py --demo
LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... python main.py --demo
```

### Fine-tuning (SFT + DPO)

`training/build_dataset.py` turns validated reports into two datasets: an SFT set
that teaches the four-way verdict format, and a DPO set whose *rejected* answer
is the collapsed single-confidence output this framework exists to avoid.

```bash
python main.py --drug SHP099 --target SHP2 --json > reports/shp099.json
python training/build_dataset.py reports/*.json --out training/data

# smoke run: a tiny local model, CPU, seconds - proves the pipeline end to end
python training/sft_train.py --tiny --data training/data/sft.jsonl --max-steps 3
python training/dpo_train.py --tiny --data training/data/dpo.jsonl --max-steps 3

# real run: any instruction-tuned base model, LoRA, needs a GPU
python training/sft_train.py --data training/data/sft.jsonl --model <checkpoint>
python training/dpo_train.py --data training/data/dpo.jsonl --model training/checkpoints/sft
```

The GNN is trained the same way:

```bash
python training/train_gnn.py --epochs 300     # writes models/gnn_logp.pt
```

See `training/README.md`. The dataset builder is tested; the training runs are
not, and no tuned checkpoint is shipped.

---

## Folder structure

```
medscriptai-onco/
├── main.py                     # CLI entry point (--demo, --drug/--target, --json)
├── config.py                   # every setting in one place, all env-optional
├── api/
│   ├── app.py                  # FastAPI: /health /drugs /analyze
│   └── static/index.html       # single-page UI (no build step)
├── src/
│   ├── agents/
│   │   ├── planner.py          # 1. hypothesis planner
│   │   ├── evidence.py         # 2. evidence retrieval
│   │   ├── analysis.py         # 3. computational analysis
│   │   ├── critic.py           # 4. falsification / critic
│   │   ├── adaptive.py         # 5. adaptive planner (loop control)
│   │   ├── decision.py         # 6. decision agent
│   │   └── workflow.py         # the graph wiring
│   ├── core/
│   │   ├── schemas.py          # Hypothesis, EvidenceItem, UncertaintyVector, VerdictReport
│   │   ├── state.py            # shared state + audit-trail logger
│   │   ├── uncertainty.py      # the six dimensions
│   │   ├── verdict.py          # four-way policy + abstain criteria
│   │   └── graph.py            # LangGraph-shaped StateGraph (dependency-free)
│   └── tools/
│       ├── smiles.py           # molecular graph reader (atoms, rings, implicit H)
│       ├── qsar.py             # Lipinski / descriptor layer
│       ├── gnn.py              # message-passing GNN (numpy)
│       ├── docking.py          # surrogate docking score
│       ├── knowledge_graph.py  # JSON snapshot + Neo4j backend
│       ├── literature.py       # local corpus + optional Europe PMC
│       ├── drugs.py            # seed compound table
│       └── llm.py              # Sarvam AI client with silent fallback
├── scripts/
│   └── load_neo4j.py           # load the KG snapshot into a real Neo4j
├── models/                     # trained GNN checkpoint + its metadata
├── training/
│   ├── build_dataset.py        # reports -> SFT + DPO datasets
│   ├── sft_train.py            # LoRA SFT, plus --tiny for a CPU smoke run
│   ├── dpo_train.py            # DPO preference tuning, plus --tiny
│   ├── train_gnn.py            # trains the GNN on a proxy property task
│   ├── tiny_model.py           # local 2-layer model used by --tiny
│   └── README.md               # the fine-tuning plan, step by step
├── data/                       # seed knowledge: targets, drugs, assays, triples, literature
├── tests/                      # 58 pytest tests, run on both engines
├── docker-compose.yml          # local Neo4j + loader
├── .github/workflows/ci.yml    # tests on 3.11 + 3.12, both engines
└── Dockerfile
```

---

## How the pieces work

**Molecular GNN (`src/tools/gnn.py`, `gnn_torch.py`)** — atom features → two
rounds of degree-normalised neighbourhood aggregation → mean pooling → a readout
that combines the embedding with the target's pocket window. Two implementations:
a NumPy one that runs anywhere, and a **PyTorch one with a trained checkpoint**
(`models/gnn_logp.pt`, produced by `training/train_gnn.py`). The checkpoint is
trained on a proxy property (MolLogP) over a 33-molecule library — that makes the
embedding genuinely learned, but the affinity readout on top of it is still
rule-based, and the report says which one ran.

**Tool orchestration (`src/tools/react_agent.py`)** — ChemCrow-style loop: pick a
tool, record the observation, decide the next one. With an LLM configured the
model chooses; without one a deterministic priority policy does, so runs stay
reproducible. Tools: RDKit descriptors, GNN, docking, knowledge-graph lookup,
literature search. The transcript lands in `report["tool_calls"]`.

**QSAR (`src/tools/qsar.py`)** — real RDKit descriptors (MolWt, MolLogP,
HBD/HBA, TPSA, rings, rotatable bonds, QED) compared against the target's
reference window, which is derived from known inhibitors of that target. Without
RDKit it falls back to the pure-Python graph estimates, and every report records
which backend answered (`backend: rdkit | graph-estimates`).

**Docking (`src/tools/docking.py`)** — shape, hydrophobic and polarity
complementarity against the pocket profile, mapped to an estimated kcal/mol.
Deterministic, so two runs never disagree.

**Literature (`src/tools/literature.py`)** — a curated local corpus by default,
plus an optional live Europe PMC search. The live parser and its
disable-after-failure behaviour are tested against the recorded response in
`data/fixtures/`, so the code is covered even where the network is blocked.

**Knowledge graph (`src/tools/knowledge_graph.py`)** — direct and two-hop paths
between drug and target, with edge polarity. Local JSON by default; set
`NEO4J_URI` and the same interface runs Cypher instead.

**Uncertainty (`src/core/uncertainty.py`)** — six components in `[0, 1]` where
1 means unresolved, combined with configurable weights in `config.py`.

---

## API

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"drug":"SHP099","target":"SHP2","disease":"KRAS-mutant NSCLC"}'
```

```json
{
  "verdict": "SUPPORTED",
  "overall_uncertainty": 0.16,
  "uncertainty": {"evidence_coverage": 0.0, "assay_consistency": 0.0,
                  "computational_agreement": 0.08, "graph_support": 0.35,
                  "literature_conflict": 0.33, "model_reliability": 0.45},
  "dominant_dimension": "model_reliability",
  "support_score": 0.75,
  "evidence": [ ... ],
  "computations": [ ... ],
  "abstain_reasons": [],
  "trace": [ ... ]
}
```

---

## What is real and what is a prototype (read this)

This is an honest research prototype, not a validated pipeline:

- **The GNN is trained, but not on affinity.** `models/gnn_logp.pt` is a real
  checkpoint trained by `training/train_gnn.py` on a proxy task (MolLogP, 33
  molecules, val MAE ≈ 0.7 logP units). The embedding is learned; the affinity
  readout on top of it is not. `model_reliability` still carries its 0.45 penalty
  for exactly that reason.
- **Docking is a surrogate.** No 3D structures, no search box, no force field.
- **The fine-tuning runs, but the result is not useful.** Both scripts execute
  end to end (`--tiny` mode: a 40k-parameter local model, CPU, a few steps) and
  the DPO mode implements the real preference objective. What has never happened
  is a training run on a real base model with a GPU — say "pipeline executed on a
  toy model", not "fine-tuned a medical LLM".
- **LangGraph and Neo4j are real but optional.** Both run for real when enabled;
  neither is the default, because the point of the default is a clone that works
  with no infrastructure.
- **Descriptors are graph estimates.** MW is accurate; logP and TPSA are
  fragment approximations (aspirin: 186 vs 180.16 Da).
- **The seed data is illustrative.** `data/*.json` is a small curated snapshot
  for demos; SMILES strings in `data/drugs.json` are simplified analogues and
  are labelled as such. Replace them with ChEMBL canonical SMILES and real
  assay exports before drawing conclusions.
- **The knowledge graph is a snapshot**, not a live ChEMBL/OpenTargets mirror.

None of this is a reason to trust a verdict — it is the reason the framework
can say `ABSTAIN`.

---

## Roadmap

- [x] Trained GNN checkpoint (torch) behind the same `predict_binding()` signature
- [~] Trained on a proxy property; a real affinity head needs assay data
- [ ] Real docking: AutoDock Vina or DiffDock adapter
- [ ] Live ingestion: ChEMBL, OpenTargets, PubMed/Europe PMC
- [~] SFT + DPO pipeline built (`training/`); no checkpoint trained yet
- [x] LangSmith / Langfuse tracing for every agent hop
- [x] Real LangGraph engine (`USE_LANGGRAPH=1`) and Neo4j backend
- [ ] Calibration study: does a low uncertainty vector actually mean a correct call?

---

## Tests

```bash
pytest -q                        # 58 tests, no network, no keys
USE_LANGGRAPH=1 pytest -q        # same suite on the LangGraph engine
python main.py --demo
```

## License

MIT.
