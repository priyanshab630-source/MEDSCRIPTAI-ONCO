"""Agent 3 - Computational Analysis Agent.

Runs the three computational signals (QSAR profile fit, molecular GNN,
surrogate docking) and normalises them to comparable [0,1] scores so the
uncertainty model can measure *disagreement* between them.
"""

import json

import config
from src.core.schemas import ComputeResult
from src.core.state import log
from src.tools import docking, docking_run, gnn, gnn_torch, qsar, react_agent


def _target_profile(target: str) -> dict:
    path = config.DATA_DIR / "targets.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        targets = json.load(handle)
    for name, profile in targets.items():
        if name.lower() == target.lower():
            return profile
    return {}


def run(state):
    hypothesis = state["hypothesis"]
    profile = _target_profile(hypothesis.target)
    smiles = hypothesis.smiles

    computations = []
    if not smiles:
        state["computations"] = []
        return log(state, "computational_analysis",
                   "no SMILES supplied - skipping in-silico checks")

    descriptors = qsar.compute(smiles)
    qsar_score, qsar_detail = qsar.score_against_profile(descriptors, profile)
    computations.append(ComputeResult(
        name="QSAR",
        score=qsar_score,
        detail=qsar_detail or "descriptor profile comparison",
        raw=descriptors,
    ))

    gnn_score, gnn_detail = gnn.predict_binding(smiles, profile)
    computations.append(ComputeResult(
        name="MolecularGNN", score=gnn_score, detail=gnn_detail,
        raw={"embedding_dim": gnn.HIDDEN, "layers": gnn.LAYERS},
    ))

    dock = docking.score(smiles, hypothesis.target, profile.get("pocket", {}))

    # If the target has a real structure, dock this compound into it and let the
    # measured pose replace the surrogate score in the computation list.
    detail = {}
    dock_status = docking_run.availability(hypothesis.target) if smiles else {"ok": False}
    if smiles and dock_status["ok"]:
        try:
            detail = docking_run.dock_compound(smiles, hypothesis.target)
            state["docking_detail"] = detail
            log(state, "analysis_agent",
                f"docked {hypothesis.drug} into {hypothesis.target} with Vina: "
                f"{detail['best_affinity']} kcal/mol")
        except Exception as exc:                       # noqa: BLE001
            log(state, "analysis_agent", f"real docking failed: {type(exc).__name__}: {exc}")
    elif smiles:
        log(state, "analysis_agent",
            f"real docking skipped - {dock_status['reason']}; using the conformer surrogate")
    computations.append(ComputeResult(
        name="Docking",
        score=dock["score01"],
        detail=f"{dock['detail']} -> {dock['kcal_per_mol']} kcal/mol (surrogate)",
        raw=dock,
    ))

    # Measured epistemic uncertainty from the trained GNN (MC dropout).
    measured = None
    try:
        measured = gnn_torch.epistemic_uncertainty(smiles)
    except Exception as exc:                      # noqa: BLE001 - torch optional
        measured = None
    state["model_reliability_measured"] = measured

    # ChemCrow-style orchestration: pick tools, record observations.
    orchestration = react_agent.run(hypothesis, profile)
    state["tool_calls"] = orchestration.get("steps", [])
    state["tool_mode"] = orchestration.get("mode", "deterministic")

    state["computations"] = computations
    scores = ", ".join(f"{c.name}={c.score:.2f}" for c in computations)
    reliability = ("MC-dropout uncertainty "
                   f"{measured:.2f}" if measured is not None else "no checkpoint")
    return log(state, "computational_analysis",
               f"{scores}; {len(state['tool_calls'])} tool call(s) via "
               f"{state['tool_mode']} policy; {reliability}")
