"""Loading, validating and splitting the bank churn dataset."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Exited"

# Row and customer identifiers. They say nothing about a customer's behaviour, so a model that
# used them could only memorise individual rows (Surname) or pick up artefacts of how the file
# was ordered (RowNumber, CustomerId). None of them would generalise to new customers.
IDENTIFIER_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

DEFAULT_DATA_PATH = Path("data/churn_modelling.csv")


@dataclass(frozen=True)
class Split:
    """Stratified train, validation and test partitions of the dataset."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Fail unless every column is present and has no missing values."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    blank = [column for column in columns if frame[column].isna().any()]
    if blank:
        raise ValueError(f"missing values in columns: {', '.join(blank)}")


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the CSV and return the feature columns and the 0/1 churn target.

    Identifier columns and any other extra columns are dropped.
    """
    frame = pd.read_csv(path)
    require_columns(frame, FEATURES + [TARGET])
    y = frame[TARGET].astype(int)
    if not set(y.unique()) <= {0, 1}:
        raise ValueError(f"{TARGET} must contain only 0 and 1")
    return frame[FEATURES].copy(), y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> Split:
    """Split into train, validation and test sets, stratified on the target.

    The test set (``test_size`` of all rows) is held out for the final evaluation only. The
    validation set (``val_size`` of the remaining rows) is used to pick the decision threshold.
    """
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size, stratify=y_rest, random_state=seed
    )
    return Split(X_train, y_train, X_val, y_val, X_test, y_test)
