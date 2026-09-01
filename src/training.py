"""
Training orchestration for CreditRisk AI.

Production workflow (HMEQ)
--------------------------
``train_and_persist()`` now expects a real dataset.  When a dataset path
is provided, it delegates to ``load_hmeq`` + ``engineer_features``
and prints a data validation summary before training.

If no dataset path is provided and ``allow_demo_fallback=True``, the
legacy synthetic dataset is used (Streamlit demo mode only).

Validation summary
------------------
The pipeline prints a compact summary before training:

  [DATA] Shape           : 5960 rows × 20 cols
  [DATA] Target dist.    : BAD=0  4771 (80.05%) | BAD=1  1189 (19.95%)
  [DATA] Missing values  : debtinc 21.26% | derog 11.88% | ...
  [DATA] Duplicates      : 0
  [FEAT] Engineered      : loan_to_value, mort_to_value, ...
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

from src.dataset_loader import generate_data_quality_report, load_hmeq
from src.features import ENGINEERED_FEATURE_NAMES, engineer_features
from src.data import load_or_create_dataset
from src.modeling import save_artifacts, train_and_select_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Production default — HMEQ dataset
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "hmeq.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Legacy demo default (kept for Streamlit demo mode backward compatibility)
_LEGACY_DEMO_PATH = PROJECT_ROOT / "data" / "demo_loan_data.csv"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_validation_summary(df_raw, df_enriched) -> None:
    """Print a compact data validation summary to stdout."""
    report = generate_data_quality_report(df_raw)

    print("\n" + "=" * 60)
    print("  DATA VALIDATION SUMMARY")
    print("=" * 60)

    # Shape
    print(f"  [DATA] Shape           : {report['n_rows']} rows × {report['n_cols']} cols")

    # Target distribution
    td = report.get("target_distribution", {})
    if td:
        bad0 = td["bad_0_non_bad_loan"]
        bad1 = td["bad_1_defaulted_loan"]
        pct1 = td["bad_1_pct"]
        pct0 = round(100 - pct1, 2)
        print(
            f"  [DATA] Target dist.    : "
            f"BAD=0 (non-bad) {bad0:>5} ({pct0:.2f}%) | "
            f"BAD=1 (default) {bad1:>5} ({pct1:.2f}%)"
        )

    # Missing values — show only columns with > 0% missing
    missing_with_gaps = [
        m for m in report["missing_values"] if m["missing_pct"] > 0
    ]
    if missing_with_gaps:
        missing_str = " | ".join(
            f"{m['column']} {m['missing_pct']}%" for m in missing_with_gaps
        )
        print(f"  [DATA] Missing values  : {missing_str}")
    else:
        print("  [DATA] Missing values  : none")

    # Duplicates
    print(f"  [DATA] Duplicates      : {report['duplicate_rows']}")

    # Engineered features actually added
    added_features = [f for f in ENGINEERED_FEATURE_NAMES if f in df_enriched.columns]
    if added_features:
        print(f"  [FEAT] Engineered      : {', '.join(added_features)}")
    else:
        print("  [FEAT] Engineered      : none")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_persist(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    demo_rows: int = 5000,
    candidate_models: Iterable[str] = ("logistic",),
    random_state: int = 42,
    allow_demo_fallback: bool = False,
) -> dict[str, Any]:
    """Load data, validate, engineer features, train, and save artifacts.

    Parameters
    ----------
    dataset_path:
        Path to the HMEQ CSV.  Defaults to ``data/raw/hmeq.csv``.
        If *None* and ``allow_demo_fallback=True``, falls back to the
        legacy synthetic demo dataset (Streamlit demo mode only).
    output_dir:
        Directory for ``model.joblib`` and ``metadata.json``.
        Defaults to ``artifacts/``.
    demo_rows:
        Row count for synthetic demo data when ``allow_demo_fallback=True``.
        Ignored for the HMEQ production path.
    candidate_models:
        Iterable of model names to evaluate.  The best by ROC-AUC is saved.
    random_state:
        Reproducible random seed for train/test split.
    allow_demo_fallback:
        Set ``True`` ONLY for the Streamlit demo mode.  When ``True`` and
        ``dataset_path`` is ``None`` or does not exist, the legacy synthetic
        dataset is used instead of raising an error.

    Returns
    -------
    dict
        Training metadata including model name, metrics, feature columns,
        row count, and data quality summary.

    Raises
    ------
    FileNotFoundError
        When the HMEQ dataset is absent and ``allow_demo_fallback=False``.
    """
    resolved_path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_ARTIFACTS_DIR

    # ------------------------------------------------------------------ #
    # Data loading — HMEQ production path vs legacy demo fallback         #
    # ------------------------------------------------------------------ #
    use_demo = allow_demo_fallback and (
        dataset_path is None or not resolved_path.exists()
    )

    if use_demo:
        # Legacy demo path — Streamlit demo mode only
        print(
            "[WARNING] Using synthetic demo dataset (allow_demo_fallback=True). "
            "This is NOT the production HMEQ workflow.",
            file=sys.stderr,
        )
        data = load_or_create_dataset(
            dataset_path=_LEGACY_DEMO_PATH,
            demo_rows=demo_rows,
            random_state=random_state,
        )
        # For the demo path, skip HMEQ-specific feature engineering and
        # validation summary; use the existing prepare_portfolio_frame path.
        from src.data import prepare_portfolio_frame
        data = prepare_portfolio_frame(data, random_state=random_state)
        data_quality_summary: dict[str, Any] = {"mode": "synthetic_demo"}
        enriched = data  # no HMEQ feature engineering on synthetic data

    else:
        # Production path — HMEQ real dataset
        data = load_hmeq(resolved_path)
        enriched = engineer_features(data)

        # Rename HMEQ target 'bad' → 'target' for the shared modeling layer
        enriched = enriched.rename(columns={"bad": "target"})

        # Print validation summary
        _print_validation_summary(data, enriched)

        # Capture quality report for metadata
        data_quality_summary = generate_data_quality_report(data)

    # ------------------------------------------------------------------ #
    # Train                                                               #
    # ------------------------------------------------------------------ #
    result = train_and_select_model(
        raw_df=enriched,
        candidate_models=candidate_models,
        random_state=random_state,
    )

    # ------------------------------------------------------------------ #
    # Save artifacts                                                      #
    # ------------------------------------------------------------------ #
    metadata: dict[str, Any] = {
        "model_name": result["model_name"],
        "metrics": result["metrics"],
        "feature_columns": result["feature_columns"],
        "rows_used": result["rows_used"],
        "dataset_path": str(resolved_path),
        "candidate_models": list(candidate_models),
        "training_failures": result["failures"],
        "data_quality_summary": data_quality_summary,
    }
    save_artifacts(result["pipeline"], metadata, output_dir=output_dir)
    return metadata
