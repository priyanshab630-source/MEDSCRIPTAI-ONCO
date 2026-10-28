"""SFT / DPO dataset construction (the part of fine-tuning we can test)."""

import json

from training.build_dataset import build, dpo_example, sft_example

REPORT = {
    "verdict": "ABSTAIN",
    "rationale": "Abstaining: the evidence base cannot resolve this claim.",
    "overall_uncertainty": 0.92,
    "dominant_dimension": "evidence_coverage",
    "support_score": 0.0,
    "uncertainty": {"evidence_coverage": 1.0, "model_reliability": 0.45},
    "abstain_reasons": ["no assay-level evidence available"],
    "evidence": [{"source": "literature", "claim": "nothing relevant", "polarity": 0,
                  "weight": 0.1}],
    "computations": [],
    "hypothesis": {"drug": "CompoundX", "target": "SHP2", "disease": ""},
}


def test_sft_example_shape():
    row = sft_example(REPORT)
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "CompoundX" in row["messages"][1]["content"]
    answer = json.loads(row["messages"][2]["content"])
    assert answer["verdict"] == "ABSTAIN"
    assert answer["abstain_reasons"]


def test_dpo_pair_prefers_structure_over_confidence():
    row = dpo_example(REPORT)
    chosen = json.loads(row["chosen"])
    rejected = json.loads(row["rejected"])
    assert chosen["verdict"] == "ABSTAIN"
    assert "confidence" in rejected and "abstain_reasons" not in rejected
    assert chosen != rejected


def test_build_writes_both_files(tmp_path):
    stats = build([REPORT, REPORT], tmp_path)
    assert stats["reports"] == 2
    sft = (tmp_path / "sft.jsonl").read_text().strip().splitlines()
    dpo = (tmp_path / "dpo.jsonl").read_text().strip().splitlines()
    assert len(sft) == len(dpo) == 2
    assert json.loads(sft[0])["metadata"]["verdict"] == "ABSTAIN"
