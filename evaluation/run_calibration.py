"""Run the calibration study end to end.

    python evaluation/run_calibration.py              # full labelled set
    python evaluation/run_calibration.py --limit 6    # quick subset
    python evaluation/run_calibration.py --out evaluation/report.json

Each hypothesis is pushed through the real six-agent pipeline; nothing is stubbed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import calibration, hypotheses  # noqa: E402
from src.agents.workflow import run_pipeline     # noqa: E402
from src.core.schemas import Hypothesis          # noqa: E402


def run(cases: list) -> list:
    results = []
    for case in cases:
        report = run_pipeline(Hypothesis(drug=case["drug"], target=case["target"],
                                         disease=case["disease"],
                                         smiles=case["smiles"]), trace=False)
        results.append({
            "drug": case["drug"], "target": case["target"],
            "label": case["label"], "verdict": report.verdict,
            "support_score": report.support_score,
            "overall_uncertainty": report.overall_uncertainty,
            "dominant_dimension": report.dominant_dimension,
            "tool_mode": report.tool_mode,
        })
        print(f"  {case['label']:8s} {case['drug']:11s} -> {case['target']:10s} "
              f"{report.verdict:13s} support {report.support_score:.2f} "
              f"uncertainty {report.overall_uncertainty:.2f}")
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibration study for the verdict model")
    parser.add_argument("--limit", type=int, default=0, help="use the first N cases")
    parser.add_argument("--out", default="evaluation/report.json")
    args = parser.parse_args(argv)

    cases = hypotheses.build()
    if args.limit:
        cases = cases[:args.limit]

    print(f"Running {len(cases)} hypotheses ({hypotheses.summary(cases)})")
    results = run(cases)
    metrics = calibration.evaluate(results)

    print("\nMetrics")
    for key, value in metrics.items():
        print(f"  {key:<32} {value}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics, "results": results}, handle, indent=2)
    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
