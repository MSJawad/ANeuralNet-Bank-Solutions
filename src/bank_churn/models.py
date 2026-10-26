"""Preprocessing and the three models that are compared."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_churn.data import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from bank_churn.nn import TorchMLPClassifier

MODEL_NAMES = ("logreg", "gboost", "mlp")


def make_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, one-hot encode categories and pass 0/1 flags through.

    Every model shares the same preprocessing, so differences in results come from the models.
    Categories never seen in training (a new country, say) encode as all zeros instead of failing.
    """
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_estimator(name: str, seed: int, device: str = "cpu", max_epochs: int = 200):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "gboost":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, early_stopping=True, random_state=seed
        )
    if name == "mlp":
        return TorchMLPClassifier(device=device, max_epochs=max_epochs, random_state=seed)
    raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")


def make_pipeline(name: str, seed: int, device: str = "cpu", max_epochs: int = 200) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor()),
            ("model", make_estimator(name, seed, device=device, max_epochs=max_epochs)),
        ]
    )
