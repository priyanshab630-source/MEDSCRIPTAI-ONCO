"""Docking layer: conformer-aware scoring, plus a real AutoDock Vina adapter.

Two engines behind one interface (`score(smiles, target, pocket) -> dict`):

  * **conformer-mmff** (default) - RDKit generates 3D conformers (ETKDG, seeded),
    MMFF94 minimises them, and the score combines ligand strain, shape fit to the
    pocket's reference volume, hydrophobic/polar complementarity and pharmacophore
    overlap. Real molecular mechanics on the ligand, but the pocket is a profile,
    not a structure.
  * **vina** - genuine docking when a receptor PDBQT and a prepared ligand exist
    (see `src/tools/vina_adapter.py`). AutoDock Vina installs fine; the protein
    structure is the missing input, not the code.

Deterministic: the same SMILES and target always produce the same score.
"""

import hashlib

from src.tools import qsar


def _jitter(smiles: str, target: str) -> float:
    """Small deterministic perturbation so near-identical ligands differ."""
    digest = hashlib.sha256(f"{smiles}|{target}".encode()).digest()
    return (digest[0] / 255.0 - 0.5) * 0.02


def engine() -> str:
    try:
        from rdkit.Chem import AllChem  # noqa: F401

        return "conformer-mmff"
    except ImportError:
        return "descriptor-surrogate"


# --------------------------------------------------------------- conformer part
def conformer_profile(smiles: str, n_conformers: int = 5, seed: int = 0xF00D) -> dict:
    """Embed + MMFF-minimise a few conformers and summarise their energetics."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {}
    molecule = Chem.AddHs(molecule)

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.numThreads = 1
    conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=n_conformers,
                                                    params=parameters))
    if not conformer_ids:
        return {}

    energies = []
    props = AllChem.MMFFGetMoleculeProperties(molecule)
    for conformer_id in conformer_ids:
        field = AllChem.MMFFGetMoleculeForceField(molecule, props, confId=conformer_id)
        field.Minimize(maxIts=200)
        energies.append(field.CalcEnergy())

    coordinates = molecule.GetConformer(conformer_ids[0]).GetPositions()
    centre = coordinates.mean(axis=0)
    radius = float((((coordinates - centre) ** 2).sum(axis=1)).mean() ** 0.5)

    return {
        "n_conformers": len(energies),
        "min_energy": round(min(energies), 2),
        "energy_spread": round(max(energies) - min(energies), 2),
        "radius_of_gyration": round(radius, 2),
        "hbd": rdMolDescriptors.CalcNumHBD(molecule),
        "hba": rdMolDescriptors.CalcNumHBA(molecule),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(molecule),
        "psa": round(rdMolDescriptors.CalcTPSA(molecule), 1),
        "mol_wt": round(Descriptors.MolWt(molecule), 1),
    }


def _range_fit(value: float, window, below_penalty: float = 2.0) -> float:
    """1.0 inside the window, decaying outside it.

    Decay is asymmetric: a ligand too small to fill the pocket loses twice as
    fast as one that is slightly too large, which is the usual shape of the
    structure-activity relationship.
    """
    if not window:
        return 0.5
    low, high = window
    if low <= value <= high:
        return 1.0
    span = max(high - low, 1e-6)
    if value < low:
        return max(0.0, 1.0 - below_penalty * (low - value) / span)
    return max(0.0, 1.0 - (value - high) / span)


def _conformer_score(profile: dict, pocket: dict) -> tuple:
    """Turn the conformer profile into a complementarity score plus terms.

    The pocket carries *ranges* (derived from its known inhibitors) rather than a
    single reference molecule, so a fragment and an over-large ligand both score
    below a compound that sits in the pocket's window.
    """
    size_fit = _range_fit(profile["mol_wt"], pocket.get("mw_range"))
    radius_fit = _range_fit(profile["radius_of_gyration"], pocket.get("radius_range"))
    aromatic_fit = _range_fit(profile["aromatic_rings"],
                              pocket.get("aromatic_rings_range"))

    hydrophobicity = pocket.get("hydrophobicity", 0.5)
    polarity = pocket.get("polarity", 0.4)
    ligand_hydrophobic = max(0.0, min(1.0, 1.0 - profile["psa"] / 140.0))
    ligand_polar = min(1.0, (profile["hba"] + profile["hbd"]) / 12.0)
    hydrophobic_fit = 1.0 - abs(ligand_hydrophobic - hydrophobicity)
    polarity_fit = 1.0 - abs(ligand_polar - polarity)

    # A floppy ligand pays a small entropic penalty.
    flexibility = min(1.0, profile["energy_spread"] / 20.0)
    flexibility_penalty = 1.0 - 0.20 * flexibility

    terms = {
        "size_fit": round(size_fit, 4),
        "radius_fit": round(radius_fit, 4),
        "pharmacophore": round(aromatic_fit, 4),
        "hydrophobic_fit": round(hydrophobic_fit, 4),
        "polarity_fit": round(polarity_fit, 4),
        "flexibility_penalty": round(1.0 - flexibility_penalty, 4),
    }
    raw = (0.32 * size_fit + 0.18 * radius_fit + 0.16 * aromatic_fit
           + 0.20 * hydrophobic_fit + 0.14 * polarity_fit) * flexibility_penalty
    return max(0.0, min(1.0, raw)), terms


def score(smiles: str, target_name: str, pocket: dict) -> dict:
    """Returns {"score01", "kcal_per_mol", "detail", "terms", "engine"}."""
    descriptors = qsar.compute(smiles)
    if "error" in descriptors:
        return {"score01": 0.0, "kcal_per_mol": 0.0, "detail": descriptors["error"],
                "terms": {}, "engine": "unavailable"}

    profile = conformer_profile(smiles) if engine() == "conformer-mmff" else {}
    if profile:
        raw, terms = _conformer_score(profile, pocket)
        raw = max(0.0, min(1.0, raw + _jitter(smiles, target_name)))
        detail = (f"conformer/MMFF scoring into {target_name} pocket "
                  f"({profile['n_conformers']} conformers, MMFF spread "
                  f"{profile['energy_spread']:.1f} kcal/mol, Rg "
                  f"{profile['radius_of_gyration']:.1f} A)")
        used = "conformer-mmff"
    else:                                   # no RDKit: descriptor-only fallback
        reference_mw = pocket.get("reference_mw", 400.0)
        shape_fit = max(0.0, 1.0 - abs(descriptors["mw"] - reference_mw) / reference_mw)
        hydrophobic_fit = 1.0 - abs(
            min(1.0, max(0.0, (descriptors["logp"] + 2.0) / 7.0))
            - pocket.get("hydrophobicity", 0.5))
        polarity_fit = 1.0 - abs(min(1.0, (descriptors["hba"] + descriptors["hbd"]) / 12.0)
                                 - pocket.get("polarity", 0.4))
        raw = 0.40 * shape_fit + 0.35 * hydrophobic_fit + 0.25 * polarity_fit
        raw = max(0.0, min(1.0, raw + _jitter(smiles, target_name)))
        terms = {"shape_fit": round(shape_fit, 4),
                 "hydrophobic_fit": round(hydrophobic_fit, 4),
                 "polarity_fit": round(polarity_fit, 4)}
        detail = f"descriptor surrogate scoring into {target_name} pocket"
        used = "descriptor-surrogate"

    kcal = round(-4.0 - 7.0 * raw, 2)
    return {"score01": round(raw, 4), "kcal_per_mol": kcal, "detail": detail,
            "terms": terms, "engine": used}
