import pytest

from bank_churn.data import (
    FEATURES,
    IDENTIFIER_COLUMNS,
    load_dataset,
    load_features,
    split_dataset,
)
from bank_churn.models import make_preprocessor


def test_load_dataset_drops_identifiers(customers_csv):
    X, y = load_dataset(customers_csv)
    assert list(X.columns) == FEATURES
    assert not set(IDENTIFIER_COLUMNS) & set(X.columns)
    assert len(X) == len(y) == 600
    assert set(y.unique()) <= {0, 1}


def test_load_dataset_reports_missing_columns(tmp_path, customers):
    path = tmp_path / "broken.csv"
    customers.drop(columns=["Age", "Exited"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns: Age, Exited"):
        load_dataset(path)


def test_load_dataset_rejects_missing_values(tmp_path, customers):
    customers.loc[3, "Balance"] = None
    path = tmp_path / "gaps.csv"
    customers.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing values"):
        load_dataset(path)


def test_load_features_does_not_need_the_target(tmp_path, customers):
    path = tmp_path / "new.csv"
    customers.drop(columns=["Exited"]).to_csv(path, index=False)
    assert list(load_features(path).columns) == FEATURES


def test_split_is_stratified_disjoint_and_reproducible(customers_csv):
    X, y = load_dataset(customers_csv)
    split = split_dataset(X, y, seed=7)
    assert (len(split.X_train), len(split.X_val), len(split.X_test)) == (384, 96, 120)
    indexes = [set(part.index) for part in (split.X_train, split.X_val, split.X_test)]
    assert sum(map(len, indexes)) == len(set().union(*indexes)) == len(X)
    for part in (split.y_train, split.y_val, split.y_test):
        assert part.mean() == pytest.approx(y.mean(), abs=0.02)
    again = split_dataset(X, y, seed=7)
    assert again.X_test.index.equals(split.X_test.index)


def test_preprocessor_output_shape(customers_csv):
    X, _ = load_dataset(customers_csv)
    transformed = make_preprocessor().fit_transform(X)
    # 6 scaled numeric + 2 binary flags + 3 countries + 2 genders
    assert transformed.shape == (600, 13)


def test_preprocessor_ignores_unseen_categories(customers_csv):
    X, _ = load_dataset(customers_csv)
    preprocessor = make_preprocessor().fit(X)
    unseen = X.head(2).copy()
    unseen["Geography"] = "Italy"
    transformed = preprocessor.transform(unseen)
    names = list(preprocessor.get_feature_names_out())
    countries = [names.index(f"Geography_{c}") for c in ("France", "Germany", "Spain")]
    assert transformed[:, countries].sum() == 0
