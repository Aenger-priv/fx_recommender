import argparse
import json
import os
from pathlib import Path

from src.evaluate import evaluate_and_report
from src.predict import load_artifacts, predict_context
from src.train_pipeline import train_and_save


def parse_args():
    """Parse CLI arguments and subcommands.

    Returns:
        argparse.Namespace: Parsed arguments including the selected subcommand.
    """
    p = argparse.ArgumentParser(description="FX Strategy Recommender")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train model")
    pt.add_argument("--data", required=True, help="Path to fx_trading_dataset.json")
    pt.add_argument(
        "--model_dir", default="models/latest", help="Where to save the model"
    )
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch_size", type=int, default=256)
    pt.add_argument(
        "--alpha",
        type=float,
        default=3.0,
        help="Recency weighting intensity (0=no weight, higher favors recent)",
    )
    pt.add_argument("--val_split", type=float, default=0.15)
    pt.add_argument("--test_split", type=float, default=0.15)
    pt.add_argument(
        "--hidden", type=int, nargs="*", default=[128, 64], help="Hidden layer sizes"
    )
    pt.add_argument("--dropout", type=float, default=0.2)
    pt.add_argument(
        "--window_days",
        type=int,
        default=0,
        help="Optional: restrict training to last N days (0 disables)",
    )

    pp = sub.add_parser("predict", help="Predict best strategy for a context")
    pp.add_argument(
        "--attributes",
        required=True,
        help="JSON string with context attributes or path to JSON file",
    )
    pp.add_argument("--model_dir", default="models/latest", help="Path to saved model")
    pp.add_argument(
        "--mc_passes", type=int, default=30, help="Monte Carlo dropout passes"
    )

    pe = sub.add_parser("eval", help="Evaluate model on test split")
    pe.add_argument("--data", required=True, help="Path to fx_trading_dataset.json")
    pe.add_argument("--model_dir", default="models/latest", help="Path to saved model")
    pe.add_argument("--val_split", type=float, default=0.15)
    pe.add_argument("--test_split", type=float, default=0.15)
    pe.add_argument(
        "--report_json", default=None, help="Optional path to save JSON report"
    )

    return p.parse_args()


def maybe_load_json_arg(s: str):
    """Interpret an argument as a JSON object or a path to a JSON file.

    Args:
        s: JSON string or a filesystem path to a JSON file.

    Returns:
        dict: Parsed JSON object.
    """
    if os.path.exists(s):
        with open(s, "r") as f:
            return json.load(f)
    return json.loads(s)


def cmd_train(args):
    """Run training using arguments from the CLI."""
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    train_and_save(
        data_path=args.data,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        val_split=args.val_split,
        test_split=args.test_split,
        hidden=args.hidden,
        dropout=args.dropout,
        window_days=args.window_days,
    )


def cmd_predict(args):
    """Run a single prediction for the provided attributes and print JSON."""
    context = maybe_load_json_arg(args.attributes)
    artifacts, meta = load_artifacts(args.model_dir)
    result = predict_context(artifacts, meta, context, mc_passes=args.mc_passes)
    print(json.dumps(result, indent=2))


def main():
    """Entry point for the CLI router."""
    args = parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    elif args.cmd == "eval":
        report = evaluate_and_report(
            data_path=args.data,
            model_dir=args.model_dir,
            val_split=args.val_split,
            test_split=args.test_split,
        )
        if args.report_json:
            with open(args.report_json, "w") as f:
                json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
