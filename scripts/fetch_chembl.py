"""Pull a real bioactivity table from the ChEMBL REST API.

    python scripts/fetch_chembl.py --target PTPN11 --name chembl_shp2
    python scripts/fetch_chembl.py --target EGFR --type IC50 --limit 2000
    python scripts/fetch_chembl.py --chembl-id CHEMBL3351120 --name chembl_x

What it does, in order:

  1. resolves the target (protein name / synonym / ChEMBL id) to a `target_chembl_id`
  2. pages through `/activity.json` keeping rows with a usable standard value
  3. resolves the canonical SMILES for every compound in batches of 50
  4. writes `data/datasets/<name>.csv` and registers it in `datasets/manifest.json`
     with the target, assay type, retrieval date, row count and SHA-256

The API needs no key. If your network blocks `www.ebi.ac.uk` (this build sandbox
does), run the same command on your machine - nothing else changes.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                       # noqa: E402
from src.tools import datasets      # noqa: E402

BASE = "https://www.ebi.ac.uk/chembl/api/data"
DESTINATION = config.ROOT / "data" / "datasets"
PAGE = 1000
TIMEOUT = 60


def _get(path: str, **params) -> dict:
    response = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT,
                            headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def resolve_target(target: str = None, chembl_id: str = None) -> dict:
    """Find one target by ChEMBL id, preferred name or gene synonym."""
    if chembl_id:
        record = _get(f"target/{chembl_id}.json")
        return {"target_chembl_id": record["target_chembl_id"],
                "pref_name": record.get("pref_name", chembl_id),
                "organism": record.get("organism", "")}

    for key in ("pref_name__iexact", "target_synonym__iexact", "pref_name__icontains"):
        payload = _get("target.json", **{key: target, "limit": 5})
        hits = [t for t in payload.get("targets", [])
                if t.get("organism") == "Homo sapiens"] or payload.get("targets", [])
        if hits:
            best = hits[0]
            return {"target_chembl_id": best["target_chembl_id"],
                    "pref_name": best.get("pref_name", target),
                    "organism": best.get("organism", "")}
    raise SystemExit(f"no ChEMBL target matched {target!r}")


def fetch_activities(target_chembl_id: str, activity_type: str = "IC50",
                     limit: int = 1000) -> list:
    """Page through activities, keeping rows with a value and a unit."""
    rows, offset = [], 0
    while len(rows) < limit:
        payload = _get("activity.json", target_chembl_id=target_chembl_id,
                       standard_type=activity_type, limit=min(PAGE, limit - len(rows)),
                       offset=offset)
        batch = payload.get("activities", [])
        if not batch:
            break
        for activity in batch:
            value = activity.get("standard_value")
            smiles_id = activity.get("molecule_chembl_id")
            if not value or not smiles_id:
                continue
            rows.append({
                "molecule_chembl_id": smiles_id,
                "standard_type": activity.get("standard_type", activity_type),
                "standard_value": value,
                "standard_units": activity.get("standard_units", "nM"),
                "pchembl_value": activity.get("pchembl_value") or "",
            })
        offset += len(batch)
        if not payload.get("page_meta", {}).get("next"):
            break
    return rows


def attach_smiles(rows: list, batch_size: int = 50) -> list:
    """Resolve canonical SMILES for each compound, in batches."""
    identifiers = sorted({row["molecule_chembl_id"] for row in rows})
    smiles_by_id = {}
    for start in range(0, len(identifiers), batch_size):
        chunk = identifiers[start:start + batch_size]
        payload = _get("molecule.json", molecule_chembl_id__in=",".join(chunk),
                       limit=batch_size)
        for molecule in payload.get("molecules", []):
            structures = molecule.get("molecule_structures") or {}
            smiles = structures.get("canonical_smiles")
            if smiles:
                smiles_by_id[molecule["molecule_chembl_id"]] = smiles

    out = []
    for row in rows:
        smiles = smiles_by_id.get(row["molecule_chembl_id"])
        if smiles:
            out.append({**row, "smiles": smiles})
    return out


def write_csv(rows: list, name: str) -> Path:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / f"{name}.csv"
    columns = ["smiles", "molecule_chembl_id", "standard_type", "standard_value",
               "standard_units", "pchembl_value"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def register(name: str, path: Path, target: dict, activity_type: str,
             rows: int) -> None:
    manifest = datasets.manifest()
    entry = next((e for e in manifest["datasets"] if e["name"] == name), None)
    if entry is None:
        entry = {"name": name}
        manifest["datasets"].append(entry)
    entry.update({
        "title": f"{target['pref_name']} bioactivity ({activity_type})",
        "file": f"data/datasets/{path.name}",
        "rows": rows,
        "task": "regression",
        "smiles_column": "smiles",
        "target_column": "pchembl_value",
        "target_units": "pChEMBL (-log10 molar)",
        "target_label": f"{target['pref_name']} pChEMBL",
        "source": f"ChEMBL API ({target['target_chembl_id']})",
        "url": f"https://www.ebi.ac.uk/chembl/target_report_card/{target['target_chembl_id']}",
        "citation": ("Mendez et al. (2019) ChEMBL: towards direct deposition of "
                     "bioassay data. Nucleic Acids Res. 47:D930-D940."),
        "license": "ChEMBL data is CC BY-SA 4.0 - attribute ChEMBL and share alike.",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "downloaded_at": time.strftime("%Y-%m-%d"),
        "notes": (f"{activity_type} measurements for {target['pref_name']} "
                  f"({target['organism']}), fetched from the ChEMBL REST API."),
    })
    datasets.MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
    datasets.reload()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch a ChEMBL bioactivity table")
    parser.add_argument("--target", help="protein name or gene synonym, e.g. PTPN11")
    parser.add_argument("--chembl-id", help="target ChEMBL id, e.g. CHEMBL3351120")
    parser.add_argument("--type", default="IC50", help="IC50, Ki, Kd, EC50 …")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--name", help="local dataset name")
    args = parser.parse_args(argv)

    if not (args.target or args.chembl_id):
        parser.error("pass --target or --chembl-id")

    try:
        target = resolve_target(args.target, args.chembl_id)
    except requests.RequestException as exc:
        print(f"ChEMBL request failed: {exc}\n"
              "This network cannot reach www.ebi.ac.uk. Run the same command on a "
              "machine that can; the code path is identical.")
        return 1

    print(f"target: {target['pref_name']} ({target['target_chembl_id']}, "
          f"{target['organism']})")

    activities = fetch_activities(target["target_chembl_id"], args.type, args.limit)
    print(f"activities with a value: {len(activities)}")

    rows = attach_smiles(activities)
    print(f"with canonical SMILES:   {len(rows)}")
    if not rows:
        print("nothing usable came back - try another activity type or target")
        return 1

    name = args.name or f"chembl_{target['pref_name'].lower().replace(' ', '_')}"
    path = write_csv(rows, name)
    register(name, path, target, args.type, len(rows))

    print(f"wrote {path} ({len(rows)} rows)")
    print(f"registered '{name}' in datasets/manifest.json")
    print(f"train with: python training/train_gnn.py --dataset {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
