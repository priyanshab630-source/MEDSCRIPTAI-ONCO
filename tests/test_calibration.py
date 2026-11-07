"""Calibration metrics and the labelled hypothesis set."""

from evaluation import calibration, hypotheses


def _result(label, verdict, support, uncertainty):
    return {"label": label, "verdict": verdict, "support_score": support,
            "overall_uncertainty": uncertainty}


def test_detection_counts_false_support_separately():
    results = [
        _result("positive", "SUPPORTED", 0.9, 0.1),
        _result("positive", "UNCERTAIN", 0.3, 0.5),
        _result("negative", "UNCERTAIN", 0.1, 0.6),
        _result("negative", "SUPPORTED", 0.8, 0.2),
    ]
    metrics = calibration.detection(results)
    assert metrics["supported_true_pairs"] == 1
    assert metrics["supported_wrong_pairs"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["false_support_rate"] == 0.5


def test_brier_rewards_calibrated_probabilities():
    good = [_result("positive", "SUPPORTED", 0.95, 0.1),
            _result("negative", "UNCERTAIN", 0.05, 0.6)]
    bad = [_result("positive", "SUPPORTED", 0.4, 0.1),
           _result("negative", "UNCERTAIN", 0.4, 0.6)]
    assert calibration.brier(good) < calibration.brier(bad)


def test_ece_is_zero_for_perfectly_calibrated_bins():
    results = [_result("positive", "SUPPORTED", 1.0, 0.1),
               _result("negative", "UNCERTAIN", 0.0, 0.9)]
    assert calibration.expected_calibration_error(results) == 0.0


def test_auroc_uses_uncertainty():
    results = [_result("positive", "SUPPORTED", 0.9, 0.05),
               _result("positive", "SUPPORTED", 0.8, 0.10),
               _result("negative", "UNCERTAIN", 0.1, 0.60),
               _result("negative", "ABSTAIN", 0.0, 0.90)]
    assert calibration.auroc_uncertainty(results) == 1.0


def test_abstention_on_unknowns():
    results = [_result("unknown", "ABSTAIN", 0.0, 0.9),
               _result("unknown", "ABSTAIN", 0.0, 0.9),
               _result("positive", "SUPPORTED", 0.9, 0.1)]
    metrics = calibration.abstention(results)
    assert metrics["abstain_on_unknown"] == 1.0
    assert metrics["abstain_on_labelled"] == 0.0


def test_hypothesis_set_has_all_three_labels():
    cases = hypotheses.build()
    counts = hypotheses.summary(cases)
    assert counts["positive"] >= 3
    assert counts["negative"] >= 3
    assert counts["unknown"] >= 1
    assert all("label" in case for case in cases)


def test_evaluate_returns_every_metric():
    metrics = calibration.evaluate([_result("positive", "SUPPORTED", 0.9, 0.1),
                                    _result("negative", "UNCERTAIN", 0.1, 0.7)])
    for key in ("detection", "brier", "ece", "auroc_uncertainty",
                "spearman_uncertainty_vs_error", "abstention", "verdicts"):
        assert key in metrics
