"""Real-world datasets, with provenance.

Every dataset this project trains on is described in `datasets/manifest.json`:
where it came from, who to cite, the licence, the row count, the target column and
a SHA-256 so a tampered or truncated file is detected instead of trained on.

Bundled (small, reproducible offline):

  * `esol` - 1128 molecules with measured aqueous solubility (Delaney 2004)
  * `bace` - 1513 compounds with BACE-1 inhibition labels (MoleculeNet)

Fetchable on demand: `python scripts/fetch_datasets.py --fetch qm7`, or any Kaggle
dataset through the Kaggle CLI (`--kaggle owner/slug`).
"""

import csv
import hashlib
import json
import random

import config

MANIFEST = config.ROOT / "datasets" / "manifest.json"


_CACHE = {}


def manifest() -> dict:
    """The parsed manifest (cached; call reload() after writing to it)."""
    if "data" not in _CACHE:
        _CACHE["data"] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return _CACHE["data"]


def reload() -> dict:
    _CACHE.clear()
    return manifest()


def entries() -> list:
    return manifest().get("datasets", [])


def get(name: str) -> dict:
    for entry in entries():
        if entry["name"].lower() == name.lower():
            return entry
    raise KeyError(f"unknown dataset: {name} (have: "
                   f"{', '.join(e['name'] for e in entries())})")


def path_for(name: str):
    return config.ROOT / get(name)["file"]


def is_available(name: str) -> bool:
    try:
        return path_for(name).is_file()
    except KeyError:
        return False


def sha256_of(name: str) -> str:
    return hashlib.sha256(path_for(name).read_bytes()).hexdigest()


def verify(name: str) -> dict:
    """Check the file against the manifest hash. Returns a report, never raises."""
    entry = get(name)
    path = path_for(name)
    if not path.is_file():
        return {"name": name, "ok": False, "reason": "file missing",
                "hint": f"python scripts/fetch_datasets.py --fetch {name}"}
    if not entry.get("sha256"):
        return {"name": name, "ok": True, "reason": "no checksum recorded"}
    actual = sha256_of(name)
    if actual == entry["sha256"]:
        return {"name": name, "ok": True, "reason": "checksum matches",
                "sha256": actual}

    # Git may convert line endings (autocrlf on Windows, or a checkout hook).
    # Compare the normalised content before calling it tampering.
    normalised = hashlib.sha256(
        path_for(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if normalised == entry["sha256"]:
        return {"name": name, "ok": True,
                "reason": "checksum matches after newline normalisation",
                "sha256": actual}

    return {"name": name, "ok": False,
            "reason": f"checksum mismatch: {actual[:16]} != {entry['sha256'][:16]}",
            "sha256": actual}


def load(name: str, drop_invalid: bool = True) -> list:
    """Rows as [{"smiles": ..., "target": float, "label": ...}]."""
    entry = get(name)
    path = path_for(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing - run: python scripts/fetch_datasets.py --fetch {name}")

    rows, dropped = [], 0
    with open(path, encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            smiles = (record.get(entry["smiles_column"]) or "").strip()
            raw_target = (record.get(entry["target_column"]) or "").strip()
            if not smiles or raw_target == "":
                dropped += 1
                continue
            try:
                target = float(raw_target)
            except ValueError:
                dropped += 1
                continue

            if drop_invalid and not _parses(smiles):
                dropped += 1
                continue

            row = {"smiles": smiles, "target": target}
            if entry.get("task") == "classification":
                row["label"] = int(target)
            rows.append(row)

    if not rows:
        raise ValueError(f"{name}: no usable rows after validation")
    if dropped:
        rows[0]["_dropped"] = dropped
    return rows


def _parses(smiles: str) -> bool:
    """RDKit validation when available, cheap sanity check otherwise."""
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return any(character.isalpha() for character in smiles)


def split(rows: list, val_fraction: float = 0.15, test_fraction: float = 0.15,
          seed: int = 13) -> tuple:
    """Deterministic three-way split (shuffle with a fixed seed)."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    n_test = max(1, int(total * test_fraction))
    n_val = max(1, int(total * val_fraction))
    return (shuffled[n_test + n_val:], shuffled[n_test:n_test + n_val],
            shuffled[:n_test])


def summary() -> list:
    """Everything the API and the dashboard need to show provenance."""
    out = []
    for entry in entries():
        available = is_available(entry["name"])
        item = {
            "name": entry["name"],
            "title": entry.get("title", entry["name"]),
            "task": entry.get("task"),
            "rows": entry.get("rows"),
            "target": entry.get("target_label", entry.get("target_column")),
            "units": entry.get("target_units", ""),
            "source": entry.get("source"),
            "url": entry.get("url"),
            "citation": entry.get("citation"),
            "license": entry.get("license"),
            "notes": entry.get("notes", ""),
            "available": available,
            "file": entry["file"],
            "bytes": entry.get("bytes", 0),
            "sha256": entry.get("sha256", ""),
        }
        if available:
            try:
                loaded = load(entry["name"])
                item["loaded_rows"] = len(loaded)
                targets = [row["target"] for row in loaded]
                item["target_range"] = [round(min(targets), 3),
                                        round(max(targets), 3)]
            except Exception as exc:              # noqa: BLE001 - report, do not fail
                item["error"] = f"{type(exc).__name__}: {exc}"
        out.append(item)
    return out
