"""
Phase 2: Model Comparison and Evaluation — CreditRisk AI.

Runs a rigorous, leakage-free comparison between Logistic Regression and
XGBoost on the HMEQ Home Equity Loan dataset.

Key guarantees
--------------
1. Single shared train/test split (random_state=42, test_size=20%, stratified).
   Both models are evaluated on the same test set.
2. All leaky columns are excluded before any model sees the data.
   The feature list is printed for audit before fitting.
3. 5-fold stratified CV runs on X_train only; the test set is never touched
   during CV.
4. Imputer and scaler statistics are fitted only on training data (sklearn
   Pipeline handles this correctly, including per-fold in CV).

Usage
-----
  python scripts/compare_models.py
  python scripts/compare_models.py --dataset data/raw/hmeq.csv
  python scripts/compare_models.py --output-dir artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 output on Windows terminals (avoids UnicodeEncodeError for box chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data import NON_FEATURE_COLUMNS, prepare_features, prepare_portfolio_frame
from src.dataset_loader import load_hmeq
from src.evaluation import plot_roc_pr_curves, run_full_comparison
from src.features import engineer_features
from src.modeling import build_pipeline, fit_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "hmeq.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60)

MODEL_REGISTRY = {
    "Logistic Regression": "logistic",
    "XGBoost": "xgboost",
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: CreditRisk AI model comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to hmeq.csv (default: data/raw/hmeq.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for comparison JSON and curve PNG.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _prepare_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load, engineer, and prepare features.  Return (X, y, feature_columns)."""

    print("\n[1] Loading HMEQ dataset ...")
    raw = load_hmeq(dataset_path)
    print(f"    Raw shape: {raw.shape}")

    print("[2] Engineering features ...")
    enriched = engineer_features(raw)
    print(f"    Enriched shape: {enriched.shape}")

    # Rename BAD → target for shared modeling layer
    enriched = enriched.rename(columns={"bad": "target"})

    # Run prepare_portfolio_frame for consistency with training pipeline.
    # This injects issue_date (synthetic dates) and days_past_due (synthesised
    # from target).  Both are in NON_FEATURE_COLUMNS and will be excluded below.
    print("[3] Running prepare_portfolio_frame ...")
    prepared = prepare_portfolio_frame(enriched, random_state=RANDOM_STATE)

    print("[4] Extracting feature matrix ...")
    X, feature_columns = prepare_features(prepared)
    y = prepared["target"].astype(int)

    return X, y, feature_columns


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------


def _verify_no_leakage(feature_columns: list[str]) -> None:
    """Print feature list and explicitly verify no leaky columns are present."""
    LEAKY = {"bad", "target", "days_past_due", "loan_amnt", "issue_date",
             "predicted_default", "default_probability", "risk_segment", "expected_loss",
             "loan_status", "status"}
    leakage_found = [c for c in feature_columns if c in LEAKY]

    print("\n" + "═" * 60)
    print("  FEATURE LIST AUDIT (pre-training)")
    print("═" * 60)
    print(f"  Total features : {len(feature_columns)}")
    print(f"  Features       : {', '.join(feature_columns)}")
    print()
    if leakage_found:
        print(f"  [FAIL] Leaky columns detected: {leakage_found}")
        sys.exit(1)
    else:
        print("  [PASS] No leaky columns detected.")
        print("  [PASS] 'target' / 'bad' absent from feature matrix.")
        print("  [PASS] 'days_past_due' (synthetic from target) absent.")
        print("  [PASS] 'loan_amnt' (all-NaN demo column) absent.")
        print("  [PASS] 'issue_date' (synthetic dates) absent.")
    print("═" * 60)


# ---------------------------------------------------------------------------
# Formatted output helpers
# ---------------------------------------------------------------------------


def _print_metrics_table(comparison: dict) -> None:
    """Print a formatted side-by-side comparison table."""
    model_names = list(comparison["models"].keys())
    metrics_key = "metrics_at_0.50_threshold"

    print("\n" + "═" * 70)
    print("  MODEL COMPARISON — HOLDOUT TEST SET RESULTS")
    print("═" * 70)
    header = f"  {'Metric':<22}" + "".join(f"  {n:<22}" for n in model_names)
    print(header)
    print("  " + "-" * 66)

    metric_labels = [
        ("ROC-AUC", "roc_auc"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1"),
    ]
    for label, key in metric_labels:
        row = f"  {label:<22}"
        for name in model_names:
            val = comparison["models"][name][metrics_key][key]
            row += f"  {val:<22.4f}"
        print(row)

    print("  " + "-" * 66)
    ks_row = f"  {'KS Statistic':<22}"
    for name in model_names:
        ks = comparison["models"][name]["ks_statistic"]
        ks_row += f"  {ks:<22.4f}"
    print(ks_row)
    print("═" * 70)


def _print_cv_table(comparison: dict) -> None:
    model_names = list(comparison["models"].keys())
    print("\n" + "═" * 70)
    print("  5-FOLD STRATIFIED CV (training set only — test set not touched)")
    print("═" * 70)
    for name in model_names:
        cv = comparison["models"][name]["cross_validation"]
        folds_str = "  ".join(f"{s:.4f}" for s in cv["fold_scores"])
        print(f"  {name}")
        print(f"    Fold scores : {folds_str}")
        print(f"    Mean AUC    : {cv['mean_roc_auc']:.4f}  ±{cv['std_roc_auc']:.4f}")
    print("═" * 70)


def _print_confusion_matrices(comparison: dict) -> None:
    model_names = list(comparison["models"].keys())
    print("\n" + "═" * 70)
    print("  CONFUSION MATRICES (threshold = 0.50)")
    print("═" * 70)
    for name in model_names:
        cm = comparison["models"][name]["confusion_matrix"]
        print(f"  {name}")
        print(f"    Predicted:    Non-bad    Default")
        print(f"    Actual Non-bad   {cm['true_negatives']:>6}     {cm['false_positives']:>6}")
        print(f"    Actual Default   {cm['false_negatives']:>6}     {cm['true_positives']:>6}")
        print()
    print("═" * 70)


def _print_threshold_table(comparison: dict, model_name: str) -> None:
    rows = comparison["models"][model_name]["threshold_analysis"]
    print(f"\n  THRESHOLD ANALYSIS — {model_name}")
    print(f"  {'Threshold':>9}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}  "
          f"{'Pred Default%':>14}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print("  " + "-" * 76)
    for r in rows:
        print(
            f"  {r['threshold']:>9.2f}  {r['precision']:>9.4f}  {r['recall']:>9.4f}  "
            f"{r['f1']:>9.4f}  {r['predicted_default_rate']*100:>13.1f}%  "
            f"{r['true_positives']:>5}  {r['false_positives']:>5}  {r['false_negatives']:>5}"
        )


def _print_calibration_table(comparison: dict, model_name: str) -> None:
    rows = comparison["models"][model_name]["calibration_table"]
    print(f"\n  CALIBRATION TABLE — {model_name}")
    print(f"  {'Bin':>4}  {'Range':>14}  {'Count':>6}  "
          f"{'Mean Pred PD':>12}  {'Observed DR':>11}  {'Cal Error':>10}")
    print("  " + "-" * 66)
    for r in rows:
        print(
            f"  {r['bin']:>4}  {r['bin_range']:>14}  {r['count']:>6}  "
            f"{r['mean_predicted_pd']:>12.4f}  {r['observed_default_rate']:>11.4f}  "
            f"{r['calibration_error']:>+10.4f}"
        )


def _print_model_selection(comparison: dict) -> None:
    """Print a model selection recommendation based on actual holdout results."""
    model_names = list(comparison["models"].keys())
    scores = {n: comparison["models"][n]["metrics_at_0.50_threshold"]["roc_auc"]
              for n in model_names}
    ks_scores = {n: comparison["models"][n]["ks_statistic"] for n in model_names}
    cv_scores = {n: comparison["models"][n]["cross_validation"]["mean_roc_auc"]
                 for n in model_names}

    best_auc = max(scores, key=scores.get)
    best_ks = max(ks_scores, key=ks_scores.get)

    print("\n" + "═" * 70)
    print("  MODEL SELECTION REASONING")
    print("═" * 70)
    for name in model_names:
        print(f"  {name}")
        print(f"    Holdout ROC-AUC : {scores[name]:.4f}")
        print(f"    CV ROC-AUC      : {cv_scores[name]:.4f}  (training data only)")
        print(f"    KS Statistic    : {ks_scores[name]:.4f}")
        print()

    print("  Credit-risk interpretability note")
    print("  ----------------------------------")
    if "Logistic Regression" in scores:
        print("  Logistic Regression produces directly interpretable coefficients")
        print("  (log-odds per unit), which support regulatory explanation requirements.")
    if "XGBoost" in scores:
        print("  XGBoost is a nonlinear ensemble; higher predictive power is expected")
        print("  but requires SHAP for feature-level explanation (Phase 3).")

    lr_auc = scores.get("Logistic Regression", 0)
    xgb_auc = scores.get("XGBoost", 0)
    diff = abs(xgb_auc - lr_auc)

    print()
    if diff < 0.01:
        rec = "Logistic Regression"
        reason = (f"ROC-AUC gap is {diff:.4f} (< 0.01) — negligible predictive "
                  f"advantage does not justify sacrificing interpretability.")
    elif xgb_auc > lr_auc:
        rec = "XGBoost"
        reason = (f"ROC-AUC advantage is {diff:.4f} — meaningful improvement over "
                  f"baseline. SHAP explanation should be added (Phase 3) before "
                  f"using in production.")
    else:
        rec = "Logistic Regression"
        reason = "XGBoost did not outperform Logistic Regression on this dataset."

    print(f"  Recommended model : {rec}")
    print(f"  Reason            : {reason}")
    print("═" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    print("\n" + "═" * 60)
    print("  CREDITRISK AI — PHASE 2: MODEL COMPARISON")
    print("═" * 60)

    # ── 1. Data preparation ─────────────────────────────────────────
    X, y, feature_columns = _prepare_data(dataset_path)

    # ── 2. Leakage audit ────────────────────────────────────────────
    _verify_no_leakage(feature_columns)

    # ── 3. Single shared stratified split ──────────────────────────
    print("\n[5] Performing single stratified train/test split ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"    Train : {len(X_train)} rows  "
          f"({(y_train == 1).sum()} defaults = {(y_train == 1).mean()*100:.1f}%)")
    print(f"    Test  : {len(X_test)} rows  "
          f"({(y_test == 1).sum()} defaults = {(y_test == 1).mean()*100:.1f}%)")
    print("    Both models will be evaluated on this identical test set.")

    # ── 4. Build unfitted pipelines (shared column structure) ───────
    print("\n[6] Building pipelines ...")
    unfitted_pipelines: dict[str, object] = {}
    fitted_pipelines: dict[str, object] = {}

    for display_name, model_key in MODEL_REGISTRY.items():
        print(f"    Building {display_name} pipeline ...")
        try:
            pipeline = build_pipeline(X_train, model_key, random_state=RANDOM_STATE)
            unfitted_pipelines[display_name] = pipeline
        except Exception as exc:
            print(f"    [SKIP] {display_name} could not be built: {exc}")

    if not unfitted_pipelines:
        print("[ERROR] No models could be built. Exiting.")
        sys.exit(1)

    # ── 5. Fit on X_train ───────────────────────────────────────────
    print("\n[7] Fitting models on X_train ...")
    for display_name, pipeline in unfitted_pipelines.items():
        print(f"    Fitting {display_name} ...")
        from sklearn.base import clone
        fitted = fit_pipeline(clone(pipeline), X_train, y_train)
        fitted_pipelines[display_name] = fitted
        print(f"    {display_name} fitted.")

    # ── 6. Full evaluation ──────────────────────────────────────────
    print("\n[8] Running evaluation ...")
    comparison = run_full_comparison(
        models=fitted_pipelines,
        cv_pipelines=unfitted_pipelines,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        thresholds=THRESHOLDS,
        cv_folds=CV_FOLDS,
        random_state=RANDOM_STATE,
    )
    comparison["feature_columns"] = feature_columns

    # ── 7. Save JSON artifact ───────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_comparison.json"

    # Convert any numpy types for JSON serialisation
    def _serialisable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    json_path.write_text(
        json.dumps(comparison, indent=2, default=_serialisable),
        encoding="utf-8",
    )
    print(f"\n  Comparison JSON saved → {json_path}")

    # ── 8. Save ROC + PR curve plot ─────────────────────────────────
    curves_path = output_dir / "roc_pr_curves.png"
    plot_roc_pr_curves(comparison, curves_path)

    # ── 9. Print results ────────────────────────────────────────────
    _print_metrics_table(comparison)
    _print_cv_table(comparison)
    _print_confusion_matrices(comparison)

    print("\n" + "═" * 70)
    print("  THRESHOLD ANALYSIS")
    print("═" * 70)
    for name in comparison["models"]:
        _print_threshold_table(comparison, name)

    print("\n" + "═" * 70)
    print("  CALIBRATION TABLES")
    print("  (mean predicted PD vs observed default rate per probability bin)")
    print("═" * 70)
    print("  A well-calibrated model has calibration_error close to 0.0.")
    print("  Note: raw model probabilities are NOT regulatory-grade PDs.")
    for name in comparison["models"]:
        _print_calibration_table(comparison, name)

    _print_model_selection(comparison)

    print(f"\n  Artifacts saved to: {output_dir}/")
    print(f"    model_comparison.json")
    print(f"    roc_pr_curves.png")
    print()


if __name__ == "__main__":
    main()
