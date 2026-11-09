"""Receptor registry: which targets can actually be docked, and where their structure is.

A target can be *scored* without a structure (the conformer/MMFF surrogate in
`docking.py`), but it can only be *docked* with one. This module is the honest
switch between the two: drop `data/receptors/<TARGET>.pdb` plus a
`<TARGET>.box.json` (search box) and the pipeline runs real AutoDock Vina for every
compound analysed against that target.

    data/receptors/SHP2.pdb          protein structure (pocket or whole chain)
    data/receptors/SHP2.box.json     {"center": [x, y, z], "size": [22, 22, 22]}

The bundled `demo_4ajl` entry is a real PDBbind complex used to exercise the path
end to end; it is a demonstration target, not SHP2.
"""

import json
from pathlib import Path

import config

RECEPTOR_DIR = config.DATA_DIR / "receptors"


def _pdb_path(target: str) -> Path:
    return RECEPTOR_DIR / f"{target.lower()}.pdb"


def _box_path(target: str) -> Path:
    return RECEPTOR_DIR / f"{target.lower()}.box.json"


def available(target: str) -> bool:
    """True when a structure and a search box exist for this target."""
    return _pdb_path(target).is_file() and _box_path(target).is_file()


def structure(target: str) -> Path:
    if not available(target):
        raise FileNotFoundError(
            f"no receptor for {target}: add data/receptors/{target.lower()}.pdb "
            f"and {target.lower()}.box.json to dock it for real")
    return _pdb_path(target)


def box(target: str) -> dict:
    if not available(target):
        raise FileNotFoundError(f"no search box for {target}")
    return json.loads(_box_path(target).read_text(encoding="utf-8"))


def registry() -> list:
    """Every target with a usable structure, for the API and the dashboard."""
    entries = []
    for path in sorted(RECEPTOR_DIR.glob("*.pdb")):
        target = path.stem.upper()
        box = {}
        if _box_path(target).is_file():
            box = json.loads(_box_path(target).read_text(encoding="utf-8"))
        entries.append({"target": target, "file": str(path.relative_to(config.ROOT)),
                        "docked": available(target), "box": box})
    return entries
