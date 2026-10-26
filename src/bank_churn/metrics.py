"""Evaluation metrics and decision threshold selection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, scores) -> float:
    """Return the score cut-off that maximises F1 for the positive (churn) class.

    Rows with ``score >= threshold`` are predicted to churn. With about one churner in five
    customers, the default cut-off of 0.5 flags too few of them; maximising F1 trades precision
    against recall of the churners instead.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair has no threshold (it is the "predict nothing" end point).
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    return float(thresholds[int(np.argmax(f1))])


def evaluate_scores(y_true, scores, threshold: float) -> dict[str, float]:
    """Threshold-free ranking metrics plus classification metrics at ``threshold``."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    predicted = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
    }


def bootstrap_interval(
    y_true, scores, metric, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``metric(y_true, scores)``.

    Resamples rows with replacement, stratified by class so every resample has both classes.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        rows = np.concatenate(
            [
                rng.choice(positives, size=len(positives)),
                rng.choice(negatives, size=len(negatives)),
            ]
        )
        values[i] = metric(y_true[rows], scores[rows])
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)
