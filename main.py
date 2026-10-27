"""MEDSCRIPTAI-ONCO - command line entry point.

Examples
--------
    python main.py --drug SHP099 --target SHP2 --disease "KRAS-mutant NSCLC"
    python main.py --drug SHP099 --target SHP2 --json
    python main.py --demo
"""

import argparse
import json
import sys

from src.agents.workflow import run_pipeline
from src.core.schemas import Hypothesis
from src.tools import drugs as drug_lookup


def build_hypothesis(args) -> Hypothesis:
    smiles = args.smiles
    if not smiles:
        record = drug_lookup.get(args.drug)
        smiles = record.get("smiles", "") if record else ""
    return Hypothesis(drug=args.drug, target=args.target,
                      disease=args.disease, smiles=smiles)


def print_report(report: dict) -> None:
    line = "=" * 74
    print(line)
    print(f"VERDICT: {report['verdict']}   "
          f"(overall uncertainty {report['overall_uncertainty']:.2f})")
    print(line)
    print(f"Rationale : {report['rationale']}")
    print(f"Dominant  : {report['dominant_dimension']}")
    print(f"Support   : {report['support_score']:.0%} of weighted evidence")

    print("\nUncertainty vector (0 = resolved, 1 = unresolved)")
    for name, value in report["uncertainty"].items():
        bar = "#" * int(round(value * 20))
        print(f"  {name:<24} {value:.2f} |{bar:<20}|")

    print("\nComputational signals")
    for item in report["computations"]:
        print(f"  {item['name']:<14} {item['score']:.2f}  {item['detail'][:70]}")

    print("\nEvidence")
    for item in report["evidence"]:
        tag = {1: "+", -1: "-", 0: "~"}[item["polarity"]]
        print(f"  [{tag}] {item['source']:<15} w={item['weight']:.2f} "
              f"{item['claim'][:70]}")

    if report["abstain_reasons"]:
        print("\nAbstain criteria triggered")
        for reason in report["abstain_reasons"]:
            print(f"  - {reason}")

    print("\nAgent trace")
    for step in report["trace"]:
        print(f"  {step['agent']:<24} {step['message']}")
    print()


DEMO = [
    ("SHP099", "SHP2", "KRAS-mutant NSCLC"),
    ("SHP099", "SHP2", "KRAS-mutant CRC"),
    ("RMC-4550", "SHP2", "KRAS-mutant CRC"),
    ("Adagrasib", "KRAS_G12C", "KRAS-mutant NSCLC"),
    ("CompoundX", "SHP2", "pancreatic cancer"),   # nothing known -> ABSTAIN
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a drug-target hypothesis")
    parser.add_argument("--drug", help="compound name, e.g. SHP099")
    parser.add_argument("--target", help="target name, e.g. SHP2")
    parser.add_argument("--disease", default="", help="indication context")
    parser.add_argument("--smiles", default="", help="override the SMILES string")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    parser.add_argument("--demo", action="store_true", help="run the built-in cases")
    args = parser.parse_args(argv)

    if args.demo:
        for drug, target, disease in DEMO:
            args.drug, args.target, args.disease = drug, target, disease
            report = run_pipeline(build_hypothesis(args))
            print_report(report.to_dict())
        return 0

    if not args.drug or not args.target:
        parser.print_help()
        return 2

    report = run_pipeline(build_hypothesis(args))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
