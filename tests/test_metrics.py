import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from bank_churn.metrics import best_f1_threshold, bootstrap_interval, evaluate_scores


def test_best_f1_threshold_separates_perfectly_ranked_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.35, 0.9])
    assert best_f1_threshold(y, scores) == pytest.approx(0.35)


def test_best_f1_threshold_balances_precision_and_recall():
    # Cut-offs at 0.9, 0.8, 0.6, 0.4, 0.2 and 0.1 give F1 0.5, 0.8, 0.667, 0.857, 0.75 and 0.667.
    y = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.1])
    assert best_f1_threshold(y, scores) == pytest.approx(0.4)


def test_evaluate_scores_known_values():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = evaluate_scores(y, scores, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    scores = y * 0.5 + rng.random(300)
    low, high = bootstrap_interval(y, scores, roc_auc_score, n_resamples=200, seed=1)
    assert low < roc_auc_score(y, scores) < high
    assert (low, high) == bootstrap_interval(y, scores, roc_auc_score, n_resamples=200, seed=1)
