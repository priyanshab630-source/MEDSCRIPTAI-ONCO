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
python training/sft_train.py --data training/data/sft.jsonl   # needs a GPU
python training/dpo_train.py --data training/data/dpo.jsonl   # needs a GPU
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
├── training/
│   ├── build_dataset.py        # reports -> SFT + DPO datasets
│   ├── sft_train.py            # LoRA SFT (needs a GPU, not run in CI)
│   ├── dpo_train.py            # preference tuning (needs a GPU, not run in CI)
│   └── README.md               # the fine-tuning plan, step by step
├── data/                       # seed knowledge: targets, drugs, assays, triples, literature
├── tests/                      # 39 pytest tests, run on both engines
├── docker-compose.yml          # local Neo4j + loader
├── .github/workflows/ci.yml    # tests on 3.11 + 3.12, both engines
└── Dockerfile
```

---

## How the pieces work

**Molecular GNN (`src/tools/gnn.py`)** — atom features → two rounds of
degree-normalised neighbourhood aggregation → mean pooling → a readout that
combines the graph embedding with the target's pocket window. NumPy only.

**QSAR (`src/tools/qsar.py`)** — molecular weight, logP, HBD/HBA, ring count,
rotatable bonds and TPSA computed from the parsed graph, then compared with the
target's reference window. No RDKit required.

**Docking (`src/tools/docking.py`)** — shape, hydrophobic and polarity
complementarity against the pocket profile, mapped to an estimated kcal/mol.
Deterministic, so two runs never disagree.

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

- **The GNN is untrained.** The message passing is real; the readout weights are
  hand-set and seeded. It demonstrates the wiring, not a learned affinity
  predictor. That is exactly why `model_reliability` carries a fixed 0.45
  penalty in the uncertainty vector — the framework prices in its own weakness.
- **Docking is a surrogate.** No 3D structures, no search box, no force field.
- **The fine-tuning is a pipeline, not a result.** `training/` builds the data and
  the training scripts are complete, but no model has been trained here and no
  checkpoint ships. Say "specified and wired", not "fine-tuned".
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

- [ ] Trained GNN checkpoint (torch) behind the same `predict_binding()` signature
- [ ] Real docking: AutoDock Vina or DiffDock adapter
- [ ] Live ingestion: ChEMBL, OpenTargets, PubMed/Europe PMC
- [~] SFT + DPO pipeline built (`training/`); no checkpoint trained yet
- [x] LangSmith / Langfuse tracing for every agent hop
- [x] Real LangGraph engine (`USE_LANGGRAPH=1`) and Neo4j backend
- [ ] Calibration study: does a low uncertainty vector actually mean a correct call?

---

## Tests

```bash
pytest -q                        # 39 tests, no network, no keys
USE_LANGGRAPH=1 pytest -q        # same suite on the LangGraph engine
python main.py --demo
```

## License

MIT.
