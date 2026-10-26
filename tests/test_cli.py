import io
import json

import pandas as pd
import pytest

from bank_churn.cli import main

# The customer scored at the end of the original course script: credit score 600, France, male,
# 40 years old, 3 years of tenure, 60000 balance, 2 products, has a card, active, 50000 salary.
COURSE_EXAMPLE = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000,
}


@pytest.fixture
def trained(tmp_path, customers_csv):
    out_dir = tmp_path / "models"
    main(["train", "--data", str(customers_csv), "--out-dir", str(out_dir), "--max-epochs", "5"])
    return out_dir


def test_train_saves_every_model(trained, capsys):
    assert sorted(p.name for p in trained.iterdir()) == [
        "gboost.joblib",
        "logreg.joblib",
        "mlp.joblib",
    ]


def test_train_a_single_model(tmp_path, customers_csv, capsys):
    out_dir = tmp_path / "one"
    main(["train", "--data", str(customers_csv), "--model", "logreg", "--out-dir", str(out_dir)])
    assert [p.name for p in out_dir.iterdir()] == ["logreg.joblib"]
    assert "Logistic regression: trained" in capsys.readouterr().out


def test_evaluate_prints_table_and_writes_outputs(trained, customers_csv, tmp_path, capsys):
    plot, metrics = tmp_path / "curves.png", tmp_path / "metrics.json"
    models = sorted(str(p) for p in trained.iterdir())
    main(
        ["evaluate", *models, "--data", str(customers_csv), "--bootstrap", "50"]
        + ["--plot", str(plot), "--json", str(metrics)]
    )
    out = capsys.readouterr().out
    assert "| Model | ROC-AUC | 95% CI | PR-AUC | 95% CI |" in out
    assert "| Model | Threshold | Precision | Recall | F1 | Accuracy |" in out
    rows = [line for line in out.splitlines() if line.startswith("| ") and "Model" not in line]
    names = ["| Logistic regression", "| Gradient boosting", "| Neural network (MLP)"]
    assert [row.split(" | ")[0] for row in rows] == names * 2
    assert "Test set: 120 customers" in out
    assert plot.stat().st_size > 10_000
    payload = json.loads(metrics.read_text())
    assert [entry["model"] for entry in payload] == ["logreg", "gboost", "mlp"]
    assert 0.5 < payload[0]["test"]["roc_auc"] <= 1


def test_evaluate_refuses_a_different_data_file(trained, tmp_path, customers):
    other = tmp_path / "other.csv"
    customers.sample(frac=1, random_state=1).to_csv(other, index=False)
    with pytest.raises(SystemExit, match="trained on a different data file"):
        main(["evaluate", str(trained / "logreg.joblib"), "--data", str(other)])


def test_predict_scores_the_course_example(trained, tmp_path):
    customers = tmp_path / "new.csv"
    pd.DataFrame([COURSE_EXAMPLE, {**COURSE_EXAMPLE, "Age": 70, "IsActiveMember": 0}]).to_csv(
        customers, index=False
    )
    output = tmp_path / "predictions.csv"
    main(
        ["predict", "--model", str(trained / "gboost.joblib")]
        + ["--input", str(customers), "--output", str(output)]
    )
    predictions = pd.read_csv(output)
    assert list(predictions.columns) == [*COURSE_EXAMPLE, "churn_probability", "churn_predicted"]
    assert predictions["churn_probability"].between(0, 1).all()
    assert set(predictions["churn_predicted"]) <= {0, 1}


def test_predict_threshold_override_and_stdout(trained, customers_csv, capsys):
    main(
        ["predict", "--model", str(trained / "mlp.joblib"), "--input", str(customers_csv)]
        + ["--threshold", "0"]
    )
    predictions = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert len(predictions) == 600
    assert (predictions["churn_predicted"] == 1).all()


def test_predict_reports_missing_columns(trained, tmp_path):
    customers = tmp_path / "bad.csv"
    pd.DataFrame([{"Age": 40}]).to_csv(customers, index=False)
    with pytest.raises(SystemExit, match="missing required columns"):
        main(["predict", "--model", str(trained / "logreg.joblib"), "--input", str(customers)])


def test_predict_reports_missing_values(trained, tmp_path):
    customers = tmp_path / "gaps.csv"
    pd.DataFrame([COURSE_EXAMPLE, {**COURSE_EXAMPLE, "Balance": None}]).to_csv(
        customers, index=False
    )
    with pytest.raises(SystemExit, match="missing values in columns: Balance"):
        main(["predict", "--model", str(trained / "mlp.joblib"), "--input", str(customers)])


def test_missing_data_file_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit, match="bank-churn: error"):
        main(["train", "--data", str(tmp_path / "nope.csv"), "--model", "logreg"])
