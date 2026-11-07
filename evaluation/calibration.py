"""Calibration and discrimination metrics, in plain Python.

What the study asks: does a *low* uncertainty actually mean a *correct* verdict?
Three separate questions, three separate numbers:

  * **discrimination** - AUROC of (1 - uncertainty) against the true label. Can
    the framework tell a real pair from a wrong one?
  * **calibration** - Brier score and expected calibration error of `support_score`
    read as P(claim is true). Is 0.7 support really right 70% of the time?
  * **abstention** - how often the framework abstains on cases it cannot know,
    versus how often it abstains on cases it could have called.

Nothing here is fitted; these are held-out metrics over the labelled set in
`evaluation/hypotheses.py`.
"""


def _pairs(results: list):
    """(probability of the claim, truth, uncertainty) for labelled cases."""
    rows = []
    for result in results:
        label = result["label"]
        if label not in ("positive", "negative"):
            continue
        rows.append({
            "probability": float(result["support_score"]),
            "truth": 1.0 if label == "positive" else 0.0,
            "uncertainty": float(result["overall_uncertainty"]),
            "verdict": result["verdict"],
        })
    return rows


def detection(results: list) -> dict:
    """How well SUPPORTED identifies true pairs.

    In an abstention-capable system, calling a wrong pair UNCERTAIN is a correct
    rejection, not an error - the dangerous mistake is calling it SUPPORTED. So
    the headline numbers are the false-support rate and balanced accuracy, not
    plain accuracy.
    """
    positives = [r for r in results if r["label"] == "positive"]
    negatives = [r for r in results if r["label"] == "negative"]

    true_positive = sum(1 for r in positives if r["verdict"] == "SUPPORTED")
    false_positive = sum(1 for r in negatives if r["verdict"] == "SUPPORTED")
    true_negative = len(negatives) - false_positive

    recall = true_positive / len(positives) if positives else 0.0
    specificity = true_negative / len(negatives) if negatives else 0.0
    precision = (true_positive / (true_positive + false_positive)
                 if (true_positive + false_positive) else 0.0)

    return {
        "positives": len(positives),
        "negatives": len(negatives),
        "supported_true_pairs": true_positive,
        "supported_wrong_pairs": false_positive,
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "false_support_rate": round(false_positive / len(negatives), 4)
        if negatives else 0.0,
        "balanced_accuracy": round(0.5 * (recall + specificity), 4),
    }


def brier(results: list) -> float:
    rows = _pairs(results)
    if not rows:
        return 0.0
    return round(sum((row["probability"] - row["truth"]) ** 2 for row in rows)
                 / len(rows), 4)


def expected_calibration_error(results: list, bins: int = 5) -> float:
    """Mean |confidence - accuracy| across probability bins."""
    rows = _pairs(results)
    if not rows:
        return 0.0

    total, error = len(rows), 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [row for row in rows
                  if low <= row["probability"] < high
                  or (index == bins - 1 and row["probability"] == 1.0)]
        if not bucket:
            continue
        mean_confidence = sum(row["probability"] for row in bucket) / len(bucket)
        mean_accuracy = sum(row["truth"] for row in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(mean_confidence - mean_accuracy)
    return round(error, 4)


def auroc_uncertainty(results: list) -> float:
    """AUROC of (1 - uncertainty) as a predictor of 'the pair is real'."""
    rows = _pairs(results)
    positives = [1.0 - row["uncertainty"] for row in rows if row["truth"] == 1.0]
    negatives = [1.0 - row["uncertainty"] for row in rows if row["truth"] == 0.0]
    if not positives or not negatives:
        return 0.0

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / (len(positives) * len(negatives)), 4)


def abstention(results: list) -> dict:
    unknown = [r for r in results if r["label"] == "unknown"]
    labelled = [r for r in results if r["label"] in ("positive", "negative")]
    return {
        "abstain_on_unknown": round(
            sum(1 for r in unknown if r["verdict"] == "ABSTAIN") / len(unknown), 4)
        if unknown else 0.0,
        "abstain_on_labelled": round(
            sum(1 for r in labelled if r["verdict"] == "ABSTAIN") / len(labelled), 4)
        if labelled else 0.0,
    }


def verdict_distribution(results: list) -> dict:
    counts = {}
    for result in results:
        counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
    return counts


def spearman_uncertainty_vs_error(results: list) -> float:
    """Rank correlation between uncertainty and being wrong."""
    rows = _pairs(results)
    if len(rows) < 3:
        return 0.0

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        rank = [0.0] * len(values)
        for position, index in enumerate(order, start=1):
            rank[index] = float(position)
        return rank

    uncertainty = ranks([row["uncertainty"] for row in rows])
    error = ranks([abs(row["probability"] - row["truth"]) for row in rows])
    n = len(rows)
    mean_u = sum(uncertainty) / n
    mean_e = sum(error) / n
    numerator = sum((u - mean_u) * (e - mean_e) for u, e in zip(uncertainty, error))
    denominator = ((sum((u - mean_u) ** 2 for u in uncertainty) ** 0.5)
                   * (sum((e - mean_e) ** 2 for e in error) ** 0.5))
    return round(numerator / denominator, 4) if denominator else 0.0


def evaluate(results: list) -> dict:
    return {
        "cases": len(results),
        "detection": detection(results),
        "brier": brier(results),
        "ece": expected_calibration_error(results),
        "auroc_uncertainty": auroc_uncertainty(results),
        "spearman_uncertainty_vs_error": spearman_uncertainty_vs_error(results),
        "abstention": abstention(results),
        "verdicts": verdict_distribution(results),
    }
