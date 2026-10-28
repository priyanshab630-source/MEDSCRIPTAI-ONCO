"""Turn validated reports into SFT and DPO training data.

The fine-tuning idea in one line: teach the model the *four-way* verdict and the
abstention habit, then prefer that behaviour over a confident-sounding answer.

    python training/build_dataset.py reports/*.json --out training/data

Each report is the JSON emitted by `python main.py --drug ... --json`.

SFT example   : the agent trace + evidence -> the structured verdict, as JSON
DPO pair      : chosen = structured verdict, rejected = a collapsed single-score
                answer ("likely effective, confidence 0.82") - the exact failure
                mode this framework exists to avoid
"""

import argparse
import json
from pathlib import Path

SYSTEM = (
    "You validate drug-target hypotheses from evidence. Reply with a JSON object "
    "containing verdict (SUPPORTED, CONTRADICTED, UNCERTAIN or ABSTAIN), rationale, "
    "the six uncertainty dimensions, and the evidence you relied on. Never invent "
    "evidence; abstain when the evidence cannot settle the claim."
)

VERDICT_KEYS = ("verdict", "rationale", "uncertainty", "dominant_dimension",
                "support_score", "abstain_reasons")


def _prompt(report: dict) -> str:
    hypothesis = report.get("hypothesis") or {}
    evidence = report.get("evidence", [])[:6]
    lines = [
        f"Hypothesis: {hypothesis.get('drug', '?')} modulates "
        f"{hypothesis.get('target', '?')} in {hypothesis.get('disease') or 'an unspecified indication'}.",
        "Evidence:",
    ]
    for item in evidence:
        lines.append(f"- [{item.get('source')}] {item.get('claim', '')[:220]}")
    computations = report.get("computations", [])
    if computations:
        lines.append("Computational signals: " + ", ".join(
            f"{c.get('name')}={c.get('score'):.2f}" for c in computations))
    lines.append("Return the verdict JSON.")
    return "\n".join(lines)


def _answer(report: dict) -> str:
    """The target the model should learn to produce."""
    payload = {key: report.get(key) for key in VERDICT_KEYS if key in report}
    payload["evidence_used"] = [
        {"source": item.get("source"), "claim": item.get("claim", "")[:160]}
        for item in report.get("evidence", [])[:4]
    ]
    return json.dumps(payload, indent=2)


def _rejected(report: dict) -> str:
    """The behaviour we want the model to move away from."""
    support = report.get("support_score", 0.5)
    return json.dumps({
        "verdict": "SUPPORTED" if support >= 0.5 else "CONTRADICTED",
        "confidence": round(0.55 + 0.4 * support, 2),
        "rationale": "Evidence looks favourable; likely an effective pair.",
        "uncertainty": {},
    }, indent=2)


def sft_example(report: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _prompt(report)},
            {"role": "assistant", "content": _answer(report)},
        ],
        "metadata": {"verdict": report.get("verdict"),
                     "uncertainty": report.get("overall_uncertainty")},
    }


def dpo_example(report: dict) -> dict:
    return {
        "prompt": f"{SYSTEM}\n\n{_prompt(report)}",
        "chosen": _answer(report),
        "rejected": _rejected(report),
        "metadata": {"verdict": report.get("verdict")},
    }


def load_reports(paths: list) -> list:
    reports = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        reports.extend(data if isinstance(data, list) else [data])
    return reports


def build(reports: list, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "sft.jsonl"
    dpo_path = out_dir / "dpo.jsonl"
    with open(sft_path, "w", encoding="utf-8") as sft, \
            open(dpo_path, "w", encoding="utf-8") as dpo:
        for report in reports:
            sft.write(json.dumps(sft_example(report)) + "\n")
            dpo.write(json.dumps(dpo_example(report)) + "\n")
    return {"reports": len(reports), "sft": str(sft_path), "dpo": str(dpo_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build SFT + DPO datasets from reports")
    parser.add_argument("reports", nargs="+", help="report JSON files")
    parser.add_argument("--out", default="training/data")
    args = parser.parse_args(argv)

    stats = build(load_reports(args.reports), Path(args.out))
    print(f"{stats['reports']} report(s) -> {stats['sft']} and {stats['dpo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
