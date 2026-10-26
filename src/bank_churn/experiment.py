"""Training, evaluation and prediction, shared by the command line interface."""

import platform
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from bank_churn.data import FEATURES, file_sha256, load_dataset, split_dataset
from bank_churn.metrics import best_f1_threshold, bootstrap_interval, evaluate_scores
from bank_churn.models import make_pipeline

DISPLAY_NAMES = {
    "logreg": "Logistic regression",
    "gboost": "Gradient boosting",
    "mlp": "Neural network (MLP)",
}


@dataclass
class TrainedModel:
    """Everything saved for one model: the fitted pipeline plus how it was produced."""

    name: str
    pipeline: object
    threshold: float
    seed: int
    data_sha256: str
    validation_metrics: dict[str, float]
    train_seconds: float
    versions: dict[str, str]

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.name, self.name)

    def scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[FEATURES])[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "TrainedModel":
        # joblib files are pickles and can run code when loaded: only load models you trained.
        model = joblib.load(path)
        if not isinstance(model, TrainedModel):
            raise ValueError(f"{path} is not a model saved by bank-churn train")
        return model


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def train_model(name: str, data_path: Path, seed: int, max_epochs: int = 200) -> TrainedModel:
    """Fit ``name`` on the training split and pick its threshold on the validation split."""
    X, y = load_dataset(data_path)
    split = split_dataset(X, y, seed=seed)
    pipeline = make_pipeline(name, seed=seed, max_epochs=max_epochs)
    started = time.perf_counter()
    pipeline.fit(split.X_train, split.y_train)
    train_seconds = time.perf_counter() - started
    val_scores = pipeline.predict_proba(split.X_val)[:, 1]
    threshold = best_f1_threshold(split.y_val, val_scores)
    return TrainedModel(
        name=name,
        pipeline=pipeline,
        threshold=threshold,
        seed=seed,
        data_sha256=file_sha256(data_path),
        validation_metrics=evaluate_scores(split.y_val, val_scores, threshold),
        train_seconds=train_seconds,
        versions=library_versions(),
    )


@dataclass
class TestResult:
    model: TrainedModel
    metrics: dict[str, float]
    roc_auc_interval: tuple[float, float]
    pr_auc_interval: tuple[float, float]
    y_test: np.ndarray
    scores: np.ndarray


def evaluate_on_test(
    models: list[TrainedModel], data_path: Path, n_resamples: int = 1000
) -> list[TestResult]:
    """Score each model on the held-out test split it was trained alongside."""
    digest = file_sha256(data_path)
    X, y = load_dataset(data_path)
    results = []
    for model in models:
        if model.data_sha256 != digest:
            raise ValueError(
                f"{model.name} was trained on a different data file; evaluate measures the "
                "held-out test split of the training data, so pass the same file"
            )
        split = split_dataset(X, y, seed=model.seed)
        y_test = split.y_test.to_numpy()
        scores = model.scores(split.X_test)
        results.append(
            TestResult(
                model=model,
                metrics=evaluate_scores(y_test, scores, model.threshold),
                roc_auc_interval=bootstrap_interval(
                    y_test, scores, roc_auc_score, n_resamples=n_resamples, seed=model.seed
                ),
                pr_auc_interval=bootstrap_interval(
                    y_test,
                    scores,
                    average_precision_score,
                    n_resamples=n_resamples,
                    seed=model.seed,
                ),
                y_test=y_test,
                scores=scores,
            )
        )
    return results


def results_tables(results: list[TestResult]) -> str:
    """Render test results as two Markdown tables: ranking quality, then decisions."""
    ranking = [
        "| Model | ROC-AUC | 95% CI | PR-AUC | 95% CI |",
        "|---|---|---|---|---|",
    ]
    decisions = [
        "| Model | Threshold | Precision | Recall | F1 | Accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        m = result.metrics
        name = result.model.display_name
        roc_low, roc_high = result.roc_auc_interval
        pr_low, pr_high = result.pr_auc_interval
        ranking.append(
            f"| {name} | {m['roc_auc']:.3f} | {roc_low:.3f}-{roc_high:.3f} "
            f"| {m['pr_auc']:.3f} | {pr_low:.3f}-{pr_high:.3f} |"
        )
        decisions.append(
            f"| {name} | {result.model.threshold:.2f} | {m['precision']:.3f} "
            f"| {m['recall']:.3f} | {m['f1']:.3f} | {m['accuracy']:.3f} |"
        )
    return "\n".join(
        ["Ranking quality (threshold-free):", "", *ranking, "", "At each model's threshold:", ""]
        + decisions
    )


def plot_curves(results: list[TestResult], path: Path) -> None:
    """Save ROC and precision-recall curves for every model side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for result in results:
        fpr, tpr, _ = roc_curve(result.y_test, result.scores)
        precision, recall, _ = precision_recall_curve(result.y_test, result.scores)
        label = result.model.display_name
        roc_ax.plot(fpr, tpr, label=f"{label} ({result.metrics['roc_auc']:.3f})", linewidth=1.8)
        pr_ax.plot(
            recall, precision, label=f"{label} ({result.metrics['pr_auc']:.3f})", linewidth=1.8
        )

    base_rate = float(np.mean(results[0].y_test))
    roc_ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="Random (0.500)")
    pr_ax.axhline(
        base_rate, color="0.6", linestyle="--", linewidth=1, label=f"Random ({base_rate:.3f})"
    )

    roc_ax.set(
        title="ROC curve", xlabel="False positive rate", ylabel="True positive rate (recall)"
    )
    pr_ax.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    for ax in (roc_ax, pr_ax):
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.3)
    roc_ax.legend(loc="lower right", frameon=False)
    # Curves fill the top of the PR plot, so the legend goes in the empty space at bottom left.
    pr_ax.legend(loc="lower left", framealpha=0.9, edgecolor="none")
    fig.suptitle(f"Held-out test set ({len(results[0].y_test)} customers, {base_rate:.1%} churned)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predict(model: TrainedModel, customers: pd.DataFrame, threshold: float | None = None):
    """Return ``customers`` with a churn probability and a 0/1 churn prediction appended."""
    cutoff = model.threshold if threshold is None else threshold
    scores = model.scores(customers)
    output = customers.copy()
    output["churn_probability"] = scores.round(4)
    output["churn_predicted"] = (scores >= cutoff).astype(int)
    return output
