"""Command line interface: ``bank-churn train | evaluate | predict``."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from bank_churn.data import DEFAULT_DATA_PATH, FEATURES, require_columns
from bank_churn.experiment import (
    TrainedModel,
    evaluate_on_test,
    plot_curves,
    predict,
    results_tables,
    train_model,
)
from bank_churn.models import MODEL_NAMES

DEFAULT_MODEL_DIR = Path("models")
DEFAULT_SEED = 42


def cmd_train(args: argparse.Namespace) -> None:
    names = MODEL_NAMES if args.model == "all" else (args.model,)
    for name in names:
        model = train_model(name, args.data, seed=args.seed, max_epochs=args.max_epochs)
        path = args.out_dir / f"{name}.joblib"
        model.save(path)
        m = model.validation_metrics
        print(
            f"{model.display_name}: trained in {model.train_seconds:.1f}s, validation "
            f"ROC-AUC {m['roc_auc']:.3f}, PR-AUC {m['pr_auc']:.3f}, "
            f"threshold {model.threshold:.2f} -> {path}"
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    paths = args.models or sorted(DEFAULT_MODEL_DIR.glob("*.joblib"))
    if not paths:
        raise SystemExit(f"no models given and none found in {DEFAULT_MODEL_DIR}/; run train first")
    order = {name: i for i, name in enumerate(MODEL_NAMES)}
    models = sorted((TrainedModel.load(p) for p in paths), key=lambda m: order.get(m.name, 99))
    results = evaluate_on_test(models, args.data, n_resamples=args.bootstrap)

    y_test = results[0].y_test
    print(results_tables(results))
    print()
    print(
        f"Test set: {len(y_test)} customers, {int(y_test.sum())} churned "
        f"({y_test.mean():.1%}). Predicting that nobody churns scores accuracy "
        f"{1 - y_test.mean():.3f}; a random ranking scores PR-AUC {y_test.mean():.3f}."
    )
    if args.plot:
        plot_curves(results, args.plot)
        print(f"Curves saved to {args.plot}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "model": r.model.name,
                "threshold": r.model.threshold,
                "seed": r.model.seed,
                "test": r.metrics,
                "roc_auc_95ci": r.roc_auc_interval,
                "pr_auc_95ci": r.pr_auc_interval,
                "validation": r.model.validation_metrics,
                "train_seconds": r.model.train_seconds,
                "versions": r.model.versions,
            }
            for r in results
        ]
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Metrics saved to {args.json}")


def cmd_predict(args: argparse.Namespace) -> None:
    model = TrainedModel.load(args.model)
    customers = pd.read_csv(args.input)
    require_columns(customers, FEATURES)
    output = predict(model, customers, threshold=args.threshold)
    if args.output:
        output.to_csv(args.output, index=False)
        print(f"Wrote {len(output)} predictions to {args.output}", file=sys.stderr)
    else:
        output.to_csv(sys.stdout, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bank-churn",
        description="Predict which bank customers will leave (churn).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="fit models on the training split and save them")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="training CSV")
    train.add_argument(
        "--model", choices=(*MODEL_NAMES, "all"), default="all", help="model to train"
    )
    train.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED, help="split and training seed")
    train.add_argument("--max-epochs", type=int, default=200, help="neural network epoch cap")
    train.set_defaults(func=cmd_train)

    evaluate = commands.add_parser("evaluate", help="score saved models on the held-out test split")
    evaluate.add_argument(
        "models", nargs="*", type=Path, help=f"saved models (default: {DEFAULT_MODEL_DIR}/*.joblib)"
    )
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="training CSV")
    evaluate.add_argument(
        "--bootstrap", type=int, default=1000, help="resamples for the confidence intervals"
    )
    evaluate.add_argument("--plot", type=Path, help="save ROC and PR curves to this image")
    evaluate.add_argument("--json", type=Path, help="save all metrics to this JSON file")
    evaluate.set_defaults(func=cmd_evaluate)

    predict_cmd = commands.add_parser("predict", help="score customers from a CSV file")
    predict_cmd.add_argument("--model", type=Path, required=True, help="saved model")
    predict_cmd.add_argument(
        "--input", type=Path, required=True, help="CSV with the feature columns"
    )
    predict_cmd.add_argument("--output", type=Path, help="output CSV (default: stdout)")
    predict_cmd.add_argument(
        "--threshold", type=float, help="override the model's saved decision threshold"
    )
    predict_cmd.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"bank-churn: error: {error}") from error
