"""Docking layer: a transparent surrogate scoring function.

A real docking engine (AutoDock Vina, DiffDock) needs structures, protonation
and a search box. For a prototype we score shape/polarity complementarity
between the ligand's descriptors and the target pocket profile, then map it to
an estimated binding free energy. Deterministic: same input, same score.
"""

import hashlib

from src.tools import qsar


def _jitter(smiles: str, target: str) -> float:
    """Small deterministic perturbation so near-identical ligands differ."""
    digest = hashlib.sha256(f"{smiles}|{target}".encode()).digest()
    return (digest[0] / 255.0 - 0.5) * 0.04


def score(smiles: str, target_name: str, pocket: dict) -> dict:
    """Returns {"score01", "kcal_per_mol", "detail", "terms"}."""
    descriptors = qsar.compute(smiles)
    if "error" in descriptors:
        return {"score01": 0.0, "kcal_per_mol": 0.0,
                "detail": descriptors["error"], "terms": {}}

    reference_mw = pocket.get("reference_mw", 400.0)
    shape_fit = max(0.0, 1.0 - abs(descriptors["mw"] - reference_mw) / reference_mw)

    pocket_hydrophobic = pocket.get("hydrophobicity", 0.5)     # 0 = polar, 1 = greasy
    ligand_hydrophobic = min(1.0, max(0.0, (descriptors["logp"] + 2.0) / 7.0))
    hydrophobic_fit = 1.0 - abs(ligand_hydrophobic - pocket_hydrophobic)

    pocket_polarity = pocket.get("polarity", 0.4)
    ligand_polarity = min(1.0, (descriptors["hba"] + descriptors["hbd"]) / 12.0)
    polarity_fit = 1.0 - abs(ligand_polarity - pocket_polarity)

    terms = {
        "shape_fit": round(shape_fit, 4),
        "hydrophobic_fit": round(hydrophobic_fit, 4),
        "polarity_fit": round(polarity_fit, 4),
    }
    raw = (0.40 * shape_fit + 0.35 * hydrophobic_fit + 0.25 * polarity_fit)
    raw = max(0.0, min(1.0, raw + _jitter(smiles, target_name)))

    # Map a perfect complementarity to about -11 kcal/mol, a poor one to -4.
    kcal = round(-4.0 - 7.0 * raw, 2)
    detail = (
        f"surrogate docking into {target_name} pocket "
        f"(shape {shape_fit:.2f}, hydrophobic {hydrophobic_fit:.2f}, "
        f"polarity {polarity_fit:.2f})"
    )
    return {"score01": round(raw, 4), "kcal_per_mol": kcal,
            "detail": detail, "terms": terms}
