"""AutoDock Vina adapter - real docking, when you supply the structures.

The prototype ships no receptor: a protein structure is a licensing question, not
a code question. Point this at your own PDBQT files and it runs the genuine Vina
scoring function and conformational search:

    from src.tools import vina_adapter
    result = vina_adapter.dock(
        receptor="data/structures/SHP2.pdbqt",
        ligand="data/structures/shp099.pdbqt",
        box={"center": [12.4, -3.1, 8.8], "size": [22, 22, 22]},
    )
    result["affinity_kcal_per_mol"]     # e.g. -9.4

Ligand preparation needs Meeko (`pip install meeko`):

    from meeko import MoleculePreparation
    MoleculePreparation().prepare_mol(rdkit_mol)   # -> PDBQT string
"""

from pathlib import Path


def available() -> bool:
    try:
        import vina  # noqa: F401

        return True
    except ImportError:
        return False


def meeko_available() -> bool:
    try:
        import meeko  # noqa: F401

        return True
    except ImportError:
        return False


def dock(receptor, ligand, box: dict, exhaustiveness: int = 8,
         seed: int = 42, cpu: int = 0) -> dict:
    """Run Vina. Returns affinity, the poses and the search metadata."""
    if not available():
        raise RuntimeError("AutoDock Vina is not installed: pip install vina")

    receptor, ligand = Path(receptor), Path(ligand)
    for path, label in ((receptor, "receptor"), (ligand, "ligand")):
        if not path.exists():
            raise RuntimeError(f"{label} PDBQT not found: {path}")

    import vina

    scorer = vina.Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=0)
    scorer.set_receptor(str(receptor))
    scorer.set_ligand_from_file(str(ligand))

    centre = box.get("center", [0, 0, 0])
    size = box.get("size", [22, 22, 22])
    scorer.compute_vina_maps(center=centre, box_size=size)

    initial = scorer.score()                    # score the input pose first
    scorer.dock(exhaustiveness=exhaustiveness, n_poses=5)
    poses = scorer.poses(n_poses=5)
    energies = scorer.energies(n_poses=5)

    best = float(energies[0][0]) if len(energies) else float(initial[0])
    return {
        "affinity_kcal_per_mol": round(best, 2),
        "initial_score_kcal_per_mol": round(float(initial[0]), 2),
        "n_poses": len(energies),
        "poses": poses,
        "box": {"center": centre, "size": size},
        "exhaustiveness": exhaustiveness,
        "seed": seed,
        "engine": "autodock-vina",
    }


def engine() -> str:
    if not available():
        return "unavailable"
    return "autodock-vina (needs receptor + ligand PDBQT)" + (
        "" if meeko_available() else "; meeko missing for ligand prep")
