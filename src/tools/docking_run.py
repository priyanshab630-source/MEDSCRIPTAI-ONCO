"""Real per-compound docking: the bridge from a hypothesis to a docked pose.

`docking.py` scores a compound against a target *profile*. This runs AutoDock Vina
against an actual structure when the target has one in `data/receptors/`, and returns
everything the report and the dashboard need: poses, the complex, the interactions
and a scene for the viewer.

Results are cached per (target, smiles) so re-running the same hypothesis is instant
and the numbers stay identical.
"""

import hashlib
import json
from pathlib import Path

import config
from src.tools import interactions, plip_adapter, receptors, scene

RUNS = config.ROOT / "runs" / "docking"
_CACHE = {}


def availability(target: str) -> dict:
    """Why real docking can or cannot run for this target.

    Three separate things have to be true, and they fail for different reasons, so
    the caller gets a reason instead of a bare False - otherwise every failure looks
    like "no receptor", which is what made docking look broken per compound.
    """
    from src.tools import vina_adapter

    if not config.DOCKING_ENABLED:
        return {"ok": False, "reason": "docking disabled (DOCKING_ENABLED=0)"}
    if not receptors.available(target):
        return {"ok": False,
                "reason": f"no structure for {target} in data/receptors/ "
                          f"(add {target.lower()}.pdb + {target.lower()}.box.json)"}
    if not vina_adapter.available():
        return {"ok": False, "reason": "AutoDock Vina is not installed (pip install vina)"}
    if not vina_adapter.meeko_available():
        return {"ok": False, "reason": "Meeko is not installed (pip install meeko)"}
    return {"ok": True, "reason": "ready"}


def available(target: str) -> bool:
    """Real docking is possible when a structure, a box and the toolchain exist."""
    return availability(target)["ok"]


def _key(target: str, smiles: str) -> str:
    digest = hashlib.sha256(f"{target}|{smiles}".encode()).hexdigest()[:12]
    return f"{target.lower()}_{digest}"


def dock_compound(smiles: str, target: str, exhaustiveness: int = 4) -> dict:
    """Dock one compound into the target's receptor. Returns a report fragment."""
    if not available(target):
        return {"engine": "unavailable", "reason":
                f"no receptor or Vina toolchain for {target}"}

    cache_key = _key(target, smiles)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    from src.tools import run_docking_helpers as helpers   # noqa: PLC0415

    out_dir = RUNS / cache_key
    out_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdb = receptors.structure(target)
    search_box = receptors.box(target)

    smiles_file = out_dir / "ligand.smi"
    smiles_file.write_text(smiles, encoding="utf-8")

    receptor_pdbqt = helpers.prepare_receptor(receptor_pdb, out_dir / "receptor.pdbqt")
    helpers.prepare_ligand(smiles_file, out_dir / "ligand.pdbqt")
    poses, crystal_score = helpers.dock(
        receptor_pdbqt, out_dir / "ligand.pdbqt", search_box["center"],
        search_box["size"][0], exhaustiveness, 9, out_dir)
    complex_pdb = helpers.build_complex(receptor_pdb, out_dir / "poses.pdbqt",
                                        out_dir / "complex.pdb")

    own = interactions.analyse(str(complex_pdb), ligand_smiles=smiles)
    result = {
        "engine": "AutoDock Vina 1.2",
        "target": target,
        "receptor": str(receptor_pdb.relative_to(config.ROOT)),
        "box": search_box,
        "poses": poses,
        "best_affinity": poses[0][0],
        "own_analysis": {"counts": own["counts"],
                         "residues": own["binding_site_residues"],
                         "hydrogen_bonds": own["hydrogen_bonds"][:12]},
        "complex": str(complex_pdb.relative_to(config.ROOT)),
    }
    if plip_adapter.available():
        plip = plip_adapter.analyse(complex_pdb, out_dir / "plip")
        result["plip"] = {"counts": plip.get("counts", {}),
                          "residues": plip.get("binding_site_residues", []),
                          "hydrogen_bonds": plip.get("hydrogen_bonds", []),
                          "hydrophobic_contacts": plip.get("hydrophobic_contacts", []),
                          "error": plip.get("error")}

    result["scene"] = scene.build_scene(complex_pdb, own)
    _CACHE[cache_key] = result
    return result
