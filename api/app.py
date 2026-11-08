"""FastAPI application for MEDSCRIPTAI-ONCO.

    uvicorn api.app:app --reload --port 8000     # then open http://localhost:8000

Routes
    GET  /                     the interactive dashboard
    GET  /health               engine, learned components, LLM mode
    GET  /drugs                seed compound table (for the form autocomplete)
    GET  /targets              target profiles
    GET  /tools                tool catalogue + which policy is driving it
    GET  /models               trained GNN + RL policy metadata
    POST /analyze              full six-agent run  -> verdict report
    POST /uncertainty          MC-dropout explorer for one SMILES
    POST /compare              several hypotheses in one request
    GET  /calibration          calibration study (cached, or computed on demand)
"""

import json
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from src.agents.workflow import run_pipeline
from src.core.engine import engine_name
from src.core.schemas import Hypothesis
from src.core import tracing
from src.tools import docking, drugs as drug_lookup, gnn_torch, llm, policy, qsar
from src.tools import react_agent

app = FastAPI(
    title="MEDSCRIPTAI-ONCO",
    description="Evidence-grounded, contradiction-aware drug-target hypothesis validation.",
    version="0.3.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_CALIBRATION_CACHE = {"report": None, "computed_at": None}


# --------------------------------------------------------------------- schemas
class HypothesisRequest(BaseModel):
    drug: str = Field(..., examples=["SHP099"])
    target: str = Field(..., examples=["SHP2"])
    disease: str = Field("", examples=["KRAS-mutant NSCLC"])
    smiles: str = ""


class UncertaintyRequest(BaseModel):
    smiles: str = Field(..., examples=["Nc1nc(N2CCNCC2)c2cc(Cl)ccc2n1"])
    n_samples: int = 25


class CompareRequest(BaseModel):
    cases: list[HypothesisRequest]


# ------------------------------------------------------------------- utilities
def _profile(target: str) -> dict:
    path = config.DATA_DIR / "targets.json"
    if not path.exists():
        return {}
    targets = json.loads(path.read_text(encoding="utf-8"))
    for name, profile in targets.items():
        if name.lower() == target.lower():
            return profile
    return {}


def _resolve_smiles(request: HypothesisRequest) -> str:
    if request.smiles:
        return request.smiles
    return drug_lookup.get(request.drug).get("smiles", "")


# ---------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": engine_name(),
        "agents": 6,
        "llm": "sarvam" if llm.available() else "rule-based fallback",
        "gnn": gnn_torch.describe(),
        "policy": policy.describe(),
        "docking": docking.engine(),
        "vina": "installed" if _vina_available() else "not installed",
        "tracing": tracing.sinks(),
    }


def _vina_available() -> bool:
    try:
        from src.tools import vina_adapter

        return vina_adapter.available()
    except Exception:                            # noqa: BLE001
        return False


@app.get("/drugs")
def list_drugs():
    return drug_lookup.all_drugs()


@app.get("/targets")
def list_targets():
    path = config.DATA_DIR / "targets.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@app.get("/tools")
def list_tools():
    return {
        "tools": policy.TOOLS,
        "priority_order": react_agent.PRIORITY,
        "active_policy": "rl-policy" if policy.load() is not None else "deterministic",
        "max_steps": getattr(config, "MAX_TOOL_STEPS", 4),
        "description": {
            "rdkit_descriptors": "RDKit physicochemical descriptors vs the target's window",
            "docking": "conformer/MMFF complementarity scoring into the pocket profile",
            "gnn_binding": "trained message-passing GNN embedding + profile readout",
            "kg_lookup": "knowledge-graph paths between drug and target",
            "literature_search": "curated literature corpus (optionally live Europe PMC)",
        },
    }


@app.get("/models")
def models():
    checkpoint = gnn_torch.CHECKPOINT
    policy_checkpoint = policy.CHECKPOINT
    return {
        "gnn": {
            "describe": gnn_torch.describe(),
            "metadata": gnn_torch.metadata(),
            "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
            "architecture": f"2-layer message passing, hidden {gnn_torch.HIDDEN}, "
                            "mean pooling, MC-dropout 0.15",
            "uncertainty_method": "Monte-Carlo dropout (25 seeded stochastic passes)",
        },
        "policy": {
            "describe": policy.describe(),
            "metadata": policy.metadata(),
            "checkpoint_bytes": policy_checkpoint.stat().st_size
            if policy_checkpoint.exists() else 0,
            "architecture": f"11 -> {policy.HIDDEN} -> 5 MLP policy network",
            "algorithm": "REINFORCE with a moving-average baseline",
            "tools": policy.TOOLS,
        },
        "docking": {"engine": docking.engine(),
                    "vina": "installed" if _vina_available() else "not installed"},
    }


@app.post("/analyze")
def analyze(request: HypothesisRequest):
    smiles = _resolve_smiles(request)
    if not smiles:
        raise HTTPException(
            status_code=422,
            detail="No SMILES for this compound: pass one explicitly in the request.",
        )

    hypothesis = Hypothesis(drug=request.drug, target=request.target,
                            disease=request.disease, smiles=smiles)
    report = run_pipeline(hypothesis)
    payload = report.to_dict()
    payload["hypothesis"] = {"drug": request.drug, "target": request.target,
                             "disease": request.disease, "smiles": smiles}
    return payload


@app.post("/uncertainty")
def uncertainty(request: UncertaintyRequest):
    """MC-dropout explorer: what the network thinks, and how much it disagrees."""
    if not request.smiles.strip():
        raise HTTPException(status_code=422, detail="smiles is required")

    descriptors = qsar.compute(request.smiles)
    distribution = gnn_torch.predict_distribution(request.smiles,
                                                  n_samples=request.n_samples)
    if distribution is None:
        return {"available": False,
                "reason": "no trained GNN checkpoint (run training/train_gnn.py)",
                "descriptors": descriptors}

    mean, std = distribution
    return {
        "available": True,
        "smiles": request.smiles,
        "n_samples": request.n_samples,
        "mean_logp": round(mean, 4),
        "std_logp": round(std, 4),
        "epistemic_uncertainty": gnn_torch.epistemic_uncertainty(
            request.smiles, request.n_samples),
        "descriptors": descriptors,
        "note": "MC dropout predicts MolLogP (the proxy training task); the spread "
                "is epistemic uncertainty, and it is seeded so it is reproducible.",
    }


@app.post("/compare")
def compare(request: CompareRequest):
    if not request.cases:
        raise HTTPException(status_code=422, detail="cases must not be empty")

    results = []
    for case in request.cases[:12]:              # keep a single request bounded
        smiles = _resolve_smiles(case)
        if not smiles:
            results.append({"drug": case.drug, "target": case.target,
                            "verdict": "SKIPPED", "reason": "no SMILES"})
            continue
        report = run_pipeline(Hypothesis(drug=case.drug, target=case.target,
                                         disease=case.disease, smiles=smiles),
                              trace=False)
        results.append({
            "drug": case.drug,
            "target": case.target,
            "verdict": report.verdict,
            "overall_uncertainty": report.overall_uncertainty,
            "support_score": report.support_score,
            "dominant_dimension": report.dominant_dimension,
            "tool_mode": report.tool_mode,
            "evidence_count": len(report.evidence),
        })
    return {"cases": results}


@app.get("/calibration")
def calibration_report(limit: int = Query(20, ge=2, le=40), refresh: bool = False):
    """Serve the stored study, or compute it now (cached afterwards)."""
    stored = config.ROOT / "evaluation" / "report.json"
    if stored.exists() and not refresh and _CALIBRATION_CACHE["report"] is None:
        _CALIBRATION_CACHE["report"] = json.loads(stored.read_text(encoding="utf-8"))
        _CALIBRATION_CACHE["computed_at"] = "from evaluation/report.json"
        return _CALIBRATION_CACHE["report"]

    if _CALIBRATION_CACHE["report"] is not None and not refresh:
        return _CALIBRATION_CACHE["report"]

    from evaluation import calibration as calibration_metrics
    from evaluation import hypotheses as hypothesis_set

    cases = hypothesis_set.build()[:limit]
    results = []
    for case in cases:
        report = run_pipeline(Hypothesis(drug=case["drug"], target=case["target"],
                                         disease=case["disease"],
                                         smiles=case["smiles"]), trace=False)
        results.append({
            "drug": case["drug"], "target": case["target"], "label": case["label"],
            "verdict": report.verdict, "support_score": report.support_score,
            "overall_uncertainty": report.overall_uncertainty,
            "dominant_dimension": report.dominant_dimension,
            "tool_mode": report.tool_mode,
        })

    payload = {"metrics": calibration_metrics.evaluate(results), "results": results,
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    _CALIBRATION_CACHE["report"] = payload
    _CALIBRATION_CACHE["computed_at"] = "computed now"
    return payload


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
