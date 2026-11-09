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
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from src.agents.workflow import run_pipeline
from src.core.engine import engine_name
from src.core.schemas import Hypothesis
from src.core import tracing
from src.tools import datasets as dataset_tools
from src.tools import render_svg
from src.tools import scene as scene_tools
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


# ---------------------------------------------------------------- docking ----
DOCKING_RUN = config.ROOT / "runs" / "4ajl"


def _docking_report() -> dict:
    path = DOCKING_RUN / "report.json"
    if not path.exists():
        return {"available": False,
                "hint": "run: python scripts/run_docking.py --receptor "
                        "data/docking/4ajl/4ajl_pocket.pdb --ligand "
                        "data/docking/4ajl/4ajl_ligand.mol2 --out runs/4ajl"}
    report = json.loads(path.read_text(encoding="utf-8"))
    report["available"] = True
    return report


@app.get("/docking")
def docking_report():
    """Docking scores, interactions and the PLIP cross-check."""
    report = _docking_report()
    if not report.get("available"):
        return report
    return {
        "available": True,
        "ligand": report.get("ligand"),
        "receptor": report.get("receptor"),
        "engine": report.get("engine"),
        "poses": report.get("poses", [])[:9],
        "crystal_pose_score": report.get("crystal_pose_score"),
        "redocking_rmsd": report.get("redocking_rmsd"),
        "box_centre": report.get("box_centre"),
        "box_size": report.get("box_size"),
        "exhaustiveness": report.get("exhaustiveness"),
        "own_analysis": {"counts": report.get("counts", {}),
                         "residues": report.get("binding_site_residues", []),
                         "hydrogen_bonds": report.get("hydrogen_bonds", []),
                         "hydrophobic_contacts": report.get("hydrophobic_contacts", []),
                         "salt_bridges": report.get("salt_bridges", [])},
        "plip": report.get("plip", {}),
    }


@app.get("/docking/view", response_class=HTMLResponse)
def docking_view():
    """Server-rendered docking page: the pose, the scores, both interaction tables.

    No JavaScript, so it prints cleanly and a headless renderer can screenshot it.
    """
    report = _docking_report()
    if not report.get("available"):
        return HTMLResponse(f"<pre>{report.get('hint', 'no run')}</pre>", status_code=404)

    scene = scene_tools.build_scene(DOCKING_RUN / "complex.pdb", report)
    svg = render_svg.render(scene)

    own = report.get("counts", {})
    plip = report.get("plip", {}) or {}
    plip_counts = plip.get("counts", {})

    def rows(items, columns):
        out = []
        for item in items[:10]:
            cells = "".join(f"<td>{item.get(column, '')}</td>" for column in columns)
            out.append(f"<tr>{cells}</tr>")
        return "".join(out) or '<tr><td colspan="4" class="muted">none</td></tr>'

    pose_rows = "".join(
        f"<tr><td>{index}</td><td class='num'>{pose[0]}</td></tr>"
        for index, pose in enumerate(report.get("poses", [])[:5], start=1))

    plip_hbonds = rows(plip.get("hydrogen_bonds", []),
                       ("residue", "donor", "acceptor", "distance"))
    plip_hydro = rows(plip.get("hydrophobic_contacts", []), ("residue", "distance"))
    own_hbonds = rows(report.get("hydrogen_bonds", []),
                      ("residue", "ligand_atom", "protein_atom", "distance"))

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Docking — {report.get('ligand')}</title><style>
body{{margin:0;background:#070d16;color:#e8eef7;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px 32px 48px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:15px;letter-spacing:.10em;text-transform:uppercase;
color:#4da3ff;margin:26px 0 10px;font-weight:600}}
.sub{{color:#8fa3bf;font-size:13px;margin-bottom:18px}}
.grid{{display:flex;gap:22px;align-items:flex-start}} .grid>.card:first-child{{flex:1.5}}
.grid>.card:last-child{{flex:1}}
.card{{background:#101c2d;border:1px solid #22334d;border-radius:12px;padding:16px;flex:1;min-width:0}}
.kpis{{display:flex;gap:12px;margin-bottom:16px}} .kpi{{flex:1}}
.kpi{{background:#101c2d;border:1px solid #22334d;border-radius:10px;padding:12px 14px}}
.kpi .label{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#8fa3bf}}
.kpi .value{{font-size:24px;font-weight:600;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#8fa3bf;
padding:6px 8px;border-bottom:1px solid #22334d}}
td{{padding:5px 8px;border-bottom:1px solid #16233a;color:#c9d6e6}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;color:#e8eef7}}
.muted{{color:#63798f}} .ok{{color:#3ddc97}}
.badge{{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;
background:#12324a;color:#7cc4ff;margin-left:6px}}
</style></head><body><div class="wrap">
<h1>Docking — {report.get('ligand')} into {Path(report.get('receptor', '')).name}</h1>
<div class="sub">{report.get('engine')} · box {report.get('box_size')} Å at
{report.get('box_centre')} · exhaustiveness {report.get('exhaustiveness')}</div>

<div class="kpis">
  <div class="kpi"><div class="label">best pose</div>
    <div class="value">{report['poses'][0][0]} <span class="muted" style="font-size:13px">kcal/mol</span></div></div>
  <div class="kpi"><div class="label">redocking RMSD</div>
    <div class="value ok">{report.get('redocking_rmsd')} <span class="muted" style="font-size:13px">Å</span></div></div>
  <div class="kpi"><div class="label">PLIP H-bonds</div>
    <div class="value">{plip_counts.get('hydrogen_bonds', 0)}</div></div>
</div>

<div class="grid">
  <div class="card">{svg}</div>
  <div class="card">
    <h2 style="margin-top:0">Poses</h2>
    <table><tr><th>pose</th><th style="text-align:right">affinity</th></tr>{pose_rows}</table>
    <p class="muted" style="font-size:12px">Crystal pose scores
      {report.get('crystal_pose_score')} kcal/mol.</p>
  </div>
</div>

<h2>Interactions — PLIP <span class="badge">reference engine</span></h2>
<div class="grid">
  <div class="card"><h2 style="margin-top:0">Hydrogen bonds</h2>
    <table><tr><th>residue</th><th>donor</th><th>acceptor</th><th>dist (Å)</th></tr>{plip_hbonds}</table></div>
  <div class="card"><h2 style="margin-top:0">Hydrophobic contacts</h2>
    <table><tr><th>residue</th><th>dist (Å)</th></tr>{plip_hydro}</table></div>
</div>

<h2>Interactions — built-in geometric analyser</h2>
<div class="card">
  <table><tr><th>residue</th><th>ligand atom</th><th>protein atom</th><th>dist (Å)</th></tr>{own_hbonds}</table>
  <p class="muted" style="font-size:12px">The built-in analyser uses plain heavy-atom cutoffs
  ({own.get('hydrogen_bonds', 0)} H-bonds, {own.get('hydrophobic_contacts', 0)} hydrophobic,
  {own.get('binding_site_residues', 0)} residues), so it is more permissive than PLIP's
  angle-aware criteria. Both are shown rather than picking the flattering one.</p>
</div>
</div></body></html>"""
    return HTMLResponse(html)


@app.get("/docking/scene")
def docking_scene():
    """Geometry for the in-browser 3D viewer."""
    report = _docking_report()
    complex_pdb = DOCKING_RUN / "complex.pdb"
    if not report.get("available") or not complex_pdb.exists():
        raise HTTPException(status_code=404, detail="no docking run available")
    return scene_tools.build_scene(complex_pdb, report)


@app.get("/datasets")
def list_datasets():
    """Every dataset the project trains on, with its provenance."""
    items = dataset_tools.summary()
    for item in items:
        if item["available"] and item.get("target"):
            try:
                item["ligand_check"] = dataset_tools.ligand_overlap(item["name"])
            except Exception as exc:              # noqa: BLE001 - never break the page
                item["ligand_check"] = {"checked": False,
                                        "reason": f"{type(exc).__name__}: {exc}"}
    return {"datasets": items, "gnn_training": gnn_torch.metadata()}


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
