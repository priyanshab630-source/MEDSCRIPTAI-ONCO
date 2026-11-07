"""The labelled hypothesis set used by the calibration study.

Three classes, all derived from the seed tables rather than invented:

  * **positive** - the drug's own target, and an assay row exists for the pair
  * **negative** - a real drug pointed at the wrong target (no assay exists)
  * **unknown**  - a compound the corpus knows nothing about

Labels are used only for scoring; the pipeline never sees them.
"""

import json

import config


def _drugs() -> dict:
    return json.loads((config.DATA_DIR / "drugs.json").read_text())


def _targets() -> dict:
    return json.loads((config.DATA_DIR / "targets.json").read_text())


def _assays() -> list:
    return json.loads((config.DATA_DIR / "assays.json").read_text())


def build() -> list:
    """[{"drug", "target", "disease", "smiles", "label"}]."""
    drugs, targets = _drugs(), _targets()
    assay_pairs = {(row["drug"], row["target"]) for row in _assays()}

    cases = []
    for name, record in drugs.items():
        target = record["target"]
        label = "positive" if (name, target) in assay_pairs else "negative"
        cases.append({
            "drug": name, "target": target, "smiles": record["smiles"],
            "disease": "KRAS-mutant NSCLC" if target != "EGFR" else "EGFR-mutant NSCLC",
            "label": label,
        })

    # negative controls: real drugs against the wrong target
    for name, record in drugs.items():
        for target in targets:
            if target == record["target"]:
                continue
            cases.append({
                "drug": name, "target": target, "smiles": record["smiles"],
                "disease": "KRAS-mutant NSCLC",
                "label": "negative",
            })

    # unknowns: nothing in the corpus mentions these
    cases.append({"drug": "CompoundX", "target": "SHP2", "smiles": "",
                  "disease": "pancreatic cancer", "label": "unknown"})
    cases.append({"drug": "CompoundY", "target": "EGFR", "smiles": "",
                  "disease": "glioma", "label": "unknown"})
    return cases


def summary(cases: list) -> dict:
    counts = {}
    for case in cases:
        counts[case["label"]] = counts.get(case["label"], 0) + 1
    return counts
