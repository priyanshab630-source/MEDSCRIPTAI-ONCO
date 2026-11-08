"""Fetch, verify and register real datasets.

    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py --verify
    python scripts/fetch_datasets.py --fetch qm7
    python scripts/fetch_datasets.py --kaggle-search "chembl bioactivity"
    python scripts/fetch_datasets.py --kaggle owner/dataset-name --name mydata \
        --smiles-col smiles --target-col activity --task regression

Provenance is written into `datasets/manifest.json` for anything fetched, so the
repository always records where its training data came from.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                       # noqa: E402
from src.tools import datasets      # noqa: E402

DESTINATION = config.ROOT / "data" / "datasets"


def _register(name: str, path: Path, **fields) -> None:
    """Add or update a manifest entry for a file on disk."""
    manifest = datasets.manifest()
    entry = next((e for e in manifest["datasets"] if e["name"] == name), None)
    if entry is None:
        entry = {"name": name, "file": f"data/datasets/{path.name}"}
        manifest["datasets"].append(entry)
    entry.update({
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "downloaded_at": time.strftime("%Y-%m-%d"),
        **fields,
    })
    datasets.MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
    datasets.reload()


def fetch(name: str) -> int:
    import requests

    entry = datasets.get(name)
    url = entry.get("url", "")
    if not url:
        print(f"{name} has no download URL in the manifest")
        return 1

    DESTINATION.mkdir(parents=True, exist_ok=True)
    destination = DESTINATION / f"{name}.csv"
    print(f"fetching {name}\n  from {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)

    _register(name, destination)
    with open(destination, encoding="utf-8", newline="") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    print(f"  saved {destination} ({destination.stat().st_size / 1024:.0f} KB, "
          f"{rows} rows)")
    print(f"  sha256 {hashlib.sha256(destination.read_bytes()).hexdigest()[:16]}")
    return 0


def verify_all() -> int:
    """Fail on a bad checksum; treat never-fetched optional datasets as skipped."""
    failures = 0
    for entry in datasets.entries():
        report = datasets.verify(entry["name"])
        optional = entry.get("downloaded_at") is None
        if report["ok"]:
            mark = "ok  "
        elif optional and report["reason"] == "file missing":
            mark = "skip"
        else:
            mark = "FAIL"
            failures += 1
        print(f"[{mark}] {entry['name']:8s} {report['reason']}")
    return 0 if failures == 0 else 1


def kaggle_search(term: str) -> int:
    """List matching Kaggle datasets (needs the Kaggle CLI + API token)."""
    try:
        result = subprocess.run(["kaggle", "datasets", "list", "-s", term,
                                 "--csv", "--page-size", "15"],
                                capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        print("kaggle CLI not found.\n  pip install kaggle\n"
              "  then save your API token to ~/.kaggle/kaggle.json "
              "(kaggle.com -> Settings -> API -> Create New Token)")
        return 1

    if result.returncode != 0:
        print(result.stderr.strip() or "kaggle search failed")
        print("\nMost failures here are credentials: check ~/.kaggle/kaggle.json")
        return result.returncode

    print(result.stdout)
    print("Pick a slug and run:\n"
          "  python scripts/fetch_datasets.py --kaggle <owner>/<slug> "
          "--name mydata --smiles-col smiles --target-col <target>")
    return 0


def kaggle_fetch(slug: str, name: str, smiles_col: str, target_col: str,
                 task: str) -> int:
    """Download a Kaggle dataset and register it with provenance."""
    if shutil.which("kaggle") is None:
        print("kaggle CLI not found.\n  pip install kaggle\n"
              "  save an API token to ~/.kaggle/kaggle.json first")
        return 1

    with tempfile.TemporaryDirectory() as temporary:
        print(f"downloading {slug} via the Kaggle CLI…")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", temporary, "--unzip"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(result.stderr.strip() or "download failed")
            print("\nCheck ~/.kaggle/kaggle.json, and that you accepted the "
                  "dataset's terms on kaggle.com.")
            return result.returncode

        candidates = sorted(Path(temporary).rglob("*.csv"))
        if not candidates:
            print("no CSV found inside the Kaggle archive")
            return 1
        source = next((c for c in candidates
                       if smiles_col in c.read_text(encoding="utf-8", errors="ignore")[:5000]),
                      candidates[0])

        DESTINATION.mkdir(parents=True, exist_ok=True)
        destination = DESTINATION / f"{name}.csv"
        shutil.copyfile(source, destination)

    with open(destination, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = sum(1 for _ in reader)
    for column in (smiles_col, target_col):
        if column not in columns:
            print(f"column {column!r} not in {columns[:8]}…")
            return 1

    _register(name, destination,
              title=f"Kaggle dataset {slug}",
              rows=rows,
              task=task,
              smiles_column=smiles_col,
              target_column=target_col,
              target_units="",
              target_label=target_col,
              source=f"Kaggle: {slug}",
              url=f"https://www.kaggle.com/datasets/{slug}",
              citation=f"Kaggle dataset {slug} - cite the dataset author, see its page",
              license="See the dataset page on Kaggle",
              notes="Fetched with the Kaggle CLI by the repository owner.")
    print(f"registered {name}: {rows} rows -> {destination}")
    print("  manifest entry updated with source, URL and checksum")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify real datasets")
    parser.add_argument("--list", action="store_true", help="show the manifest")
    parser.add_argument("--verify", action="store_true", help="check checksums")
    parser.add_argument("--fetch", metavar="NAME", help="download a manifest dataset")
    parser.add_argument("--kaggle-search", metavar="TERM")
    parser.add_argument("--kaggle", metavar="OWNER/SLUG")
    parser.add_argument("--name", help="local name for a Kaggle dataset")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--task", default="regression",
                        choices=["regression", "classification"])
    args = parser.parse_args(argv)

    if args.list:
        for item in datasets.summary():
            state = "bundled" if item["available"] else "not fetched"
            print(f"{item['name']:8s} {item['task']:14s} {str(item['rows'] or '?'):>5s} rows  "
                  f"{state:12s} {item['source']}")
            print(f"         {item['citation'][:110]}")
        print("\nKaggle datasets are supported too - see "
              "`--kaggle-search` and the note in datasets/manifest.json")
        return 0

    if args.verify:
        return verify_all()

    if args.fetch:
        return fetch(args.fetch)

    if args.kaggle_search:
        return kaggle_search(args.kaggle_search)

    if args.kaggle:
        return kaggle_fetch(args.kaggle, args.name or args.kaggle.split("/")[-1],
                            args.smiles_col, args.target_col, args.task)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
