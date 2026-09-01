"""CLI entrypoint for training the CreditRisk AI model.

Default dataset: data/raw/hmeq.csv (HMEQ Home Equity Loan dataset).

Usage examples
--------------
# Production: train on HMEQ dataset (must be placed in data/raw/hmeq.csv)
python scripts/train_model.py

# Explicit dataset path
python scripts/train_model.py --dataset data/raw/hmeq.csv

# Legacy demo mode (synthetic data, for Streamlit testing only)
python scripts/train_model.py --demo-fallback

# Both LR and XGBoost (XGBoost requires: pip install xgboost)
python scripts/train_model.py --models logistic xgboost
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import train_and_persist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train CreditRisk AI model and persist artifacts.\n"
            "Default dataset: data/raw/hmeq.csv (HMEQ Home Equity Loan)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help=(
            "Path to CSV dataset.  Defaults to data/raw/hmeq.csv. "
            "The file must exist unless --demo-fallback is set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts"),
        help="Directory to save model.joblib and metadata.json.",
    )
    parser.add_argument(
        "--demo-rows",
        type=int,
        default=5000,
        help=(
            "Row count for synthetic demo dataset when --demo-fallback "
            "is used.  Ignored for the HMEQ production path."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic"],
        help="Candidate models to evaluate. Supported: logistic, xgboost.",
    )
    parser.add_argument(
        "--demo-fallback",
        action="store_true",
        default=False,
        help=(
            "LEGACY / TESTING ONLY.  Use synthetic demo data instead of the "
            "HMEQ dataset.  Do not use this for production model evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset) if args.dataset else None
    output_dir = Path(args.output_dir)

    metadata = train_and_persist(
        dataset_path=dataset_path,
        output_dir=output_dir,
        demo_rows=args.demo_rows,
        candidate_models=args.models,
        allow_demo_fallback=args.demo_fallback,
    )

    print("Training complete.")
    print(f"Selected model : {metadata['model_name']}")
    print(f"Rows used      : {metadata['rows_used']}")
    print("Metrics:")
    print(json.dumps(metadata["metrics"], indent=2))
    print(f"Artifacts saved: {output_dir}")


if __name__ == "__main__":
    main()
