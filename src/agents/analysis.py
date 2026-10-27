"""Agent 3 - Computational Analysis Agent.

Runs the three computational signals (QSAR profile fit, molecular GNN,
surrogate docking) and normalises them to comparable [0,1] scores so the
uncertainty model can measure *disagreement* between them.
"""

import json

import config
from src.core.schemas import ComputeResult
from src.core.state import log
from src.tools import docking, gnn, qsar


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
    computations.append(ComputeResult(
        name="Docking",
        score=dock["score01"],
        detail=f"{dock['detail']} -> {dock['kcal_per_mol']} kcal/mol (surrogate)",
        raw=dock,
    ))

    state["computations"] = computations
    scores = ", ".join(f"{c.name}={c.score:.2f}" for c in computations)
    return log(state, "computational_analysis", scores)
