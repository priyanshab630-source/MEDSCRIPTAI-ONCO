"""Turn a saved ChEMBL activity export into a registered training dataset.

Use this when you already hold ChEMBL activity records - for example pulled
through the ChEMBL MCP connector (`get_bioactivity`) or saved from the REST
`/activity.json` endpoint - and want them in the same shape that
`scripts/fetch_chembl.py` produces, without a second network round trip.

    python scripts/import_chembl_activities.py activities.json \
        --target-id CHEMBL3864 --target-name "SHP2 (PTPN11)" --name chembl_shp2

The input is a JSON file holding either a list of activity records or an object
with an `activities` list. Each record needs `molecule_chembl_id`,
`canonical_smiles`, `standard_type`, `standard_relation`, `standard_value`,
`standard_units` and `pchembl_value`.

Quality filters, applied in this order (counts are printed):

  1. standard_type matches --type (default IC50)
  2. a pChEMBL value exists (ChEMBL only computes it for exact, nM-unit values)
  3. standard_relation is '=' (censored '>' / '<' rows are dropped)
  4. no data_validity_comment (ChEMBL's own "potential author/transcription
     error" flags exclude the row)
  5. a canonical SMILES exists
  6. one row per compound: repeated measurements collapse to the median pChEMBL,
     and the number of measurements is kept in `n_measurements`
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_chembl  # noqa: E402


def load_activities(path: Path) -> list:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("activities", [])
    if not isinstance(payload, list):
        raise SystemExit(f"{path}: expected a list or an object with 'activities'")
    return payload


def clean(records: list, activity_type: str = "IC50") -> tuple:
    """Apply the quality filters; return (rows, per-filter drop counts)."""
    dropped = {"wrong_type": 0, "no_pchembl": 0, "censored": 0,
               "flagged_invalid": 0, "no_smiles": 0}
    by_molecule = {}

    for record in records:
        if record.get("standard_type") != activity_type:
            dropped["wrong_type"] += 1
            continue
        if record.get("pchembl_value") in (None, ""):
            dropped["no_pchembl"] += 1
            continue
        if record.get("standard_relation") != "=":
            dropped["censored"] += 1
            continue
        if record.get("data_validity_comment"):
            dropped["flagged_invalid"] += 1
            continue
        smiles = record.get("canonical_smiles")
        if not smiles:
            dropped["no_smiles"] += 1
            continue
        molecule = record["molecule_chembl_id"]
        entry = by_molecule.setdefault(molecule, {"smiles": smiles, "pchembl": [],
                                                  "values": [], "units": set()})
        entry["pchembl"].append(float(record["pchembl_value"]))
        entry["values"].append(float(record["standard_value"]))
        entry["units"].add(record.get("standard_units") or "nM")

    rows = []
    for molecule, entry in sorted(by_molecule.items()):
        rows.append({
            "smiles": entry["smiles"],
            "molecule_chembl_id": molecule,
            "standard_type": activity_type,
            "standard_value": round(statistics.median(entry["values"]), 6),
            "standard_units": sorted(entry["units"])[0],
            "pchembl_value": round(statistics.median(entry["pchembl"]), 4),
            "n_measurements": len(entry["pchembl"]),
        })
    return rows, dropped


def write_rows(rows: list, name: str) -> Path:
    import csv
    fetch_chembl.DESTINATION.mkdir(parents=True, exist_ok=True)
    path = fetch_chembl.DESTINATION / f"{name}.csv"
    columns = ["smiles", "molecule_chembl_id", "standard_type", "standard_value",
               "standard_units", "pchembl_value", "n_measurements"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Import saved ChEMBL activities")
    parser.add_argument("activities", type=Path, help="JSON export of activity records")
    parser.add_argument("--target-id", required=True, help="e.g. CHEMBL3864")
    parser.add_argument("--target-name", required=True, help="label for the manifest")
    parser.add_argument("--organism", default="Homo sapiens")
    parser.add_argument("--type", default="IC50")
    parser.add_argument("--name", required=True, help="local dataset name")
    args = parser.parse_args(argv)

    records = load_activities(args.activities)
    rows, dropped = clean(records, args.type)

    print(f"records read:        {len(records)}")
    for reason, count in dropped.items():
        print(f"  dropped {reason:<16} {count}")
    print(f"unique compounds:    {len(rows)}")
    if not rows:
        print("nothing usable after filtering")
        return 1

    path = write_rows(rows, args.name)
    target = {"target_chembl_id": args.target_id, "pref_name": args.target_name,
              "organism": args.organism}
    fetch_chembl.register(args.name, path, target, args.type, len(rows))
    print(f"wrote {path} ({len(rows)} rows)")
    print(f"registered '{args.name}' in datasets/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
