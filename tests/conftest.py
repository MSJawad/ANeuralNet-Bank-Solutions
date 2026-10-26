import numpy as np
import pandas as pd
import pytest


def synthetic_customers(n: int, seed: int = 0) -> pd.DataFrame:
    """A small frame shaped like the real dataset, where churn depends on age and activity."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, size=n)
    active = rng.integers(0, 2, size=n)
    geography = rng.choice(["France", "Germany", "Spain"], size=n)
    logit = 0.12 * (age - 45) - 1.5 * active + 0.8 * (geography == "Germany") - 0.5
    exited = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "RowNumber": np.arange(1, n + 1),
            "CustomerId": rng.integers(15_000_000, 16_000_000, size=n),
            "Surname": rng.choice(["Hargrave", "Hill", "Onio", "Boni"], size=n),
            "CreditScore": rng.integers(350, 851, size=n),
            "Geography": geography,
            "Gender": rng.choice(["Female", "Male"], size=n),
            "Age": age,
            "Tenure": rng.integers(0, 11, size=n),
            "Balance": np.round(rng.choice([0.0, 1.0], size=n) * rng.uniform(0, 250_000, n), 2),
            "NumOfProducts": rng.integers(1, 5, size=n),
            "HasCrCard": rng.integers(0, 2, size=n),
            "IsActiveMember": active,
            "EstimatedSalary": np.round(rng.uniform(10, 200_000, size=n), 2),
            "Exited": exited,
        }
    )


@pytest.fixture
def customers() -> pd.DataFrame:
    return synthetic_customers(600)


@pytest.fixture
def customers_csv(tmp_path, customers):
    path = tmp_path / "customers.csv"
    customers.to_csv(path, index=False)
    return path
