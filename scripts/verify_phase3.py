"""
Phase 3 Verification Script — Model Explainability and Interpretability.

Usage
-----
  python scripts/verify_phase3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data import prepare_features, prepare_portfolio_frame
from src.dataset_loader import load_hmeq
from src.explainability import (
    compute_shap_explanation,
    explain_loan_application,
    extract_logistic_coefficients,
    get_preprocessed_feature_names,
    plot_global_shap_importance,
    plot_shap_summary,
    plot_xgboost_importance_comparison,
    verify_shap_reconstruction,
)
from src.features import engineer_features
from src.modeling import build_pipeline, fit_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "hmeq.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "explainability"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    print("\n" + "=" * 65)
    print("  CREDITRISK AI — PHASE 3: MODEL EXPLAINABILITY & INTERPRETABILITY")
    print("=" * 65)

    # ── 1. Data Preparation ─────────────────────────────────────────
    print("\n[1] Loading HMEQ dataset & preparing feature matrix...")
    raw = load_hmeq(DEFAULT_DATASET)
    enriched = engineer_features(raw)
    enriched = enriched.rename(columns={"bad": "target"})
    prepared = prepare_portfolio_frame(enriched, random_state=RANDOM_STATE)
    X, feature_columns = prepare_features(prepared)
    y = prepared["target"].astype(int)

    # Verify no target in feature columns
    assert "target" not in feature_columns and "bad" not in feature_columns, "Target leak!"

    # Single stratified split (identical to Phase 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Dataset loaded: {len(X)} rows | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"    Feature matrix columns: {len(feature_columns)}")

    # ── 2. Fit Pipelines ────────────────────────────────────────────
    print("\n[2] Fitting XGBoost and Logistic Regression pipelines...")
    pipe_xgb = build_pipeline(X_train, "xgboost", random_state=RANDOM_STATE)
    pipe_xgb = fit_pipeline(pipe_xgb, X_train, y_train)

    pipe_lr = build_pipeline(X_train, "logistic", random_state=RANDOM_STATE)
    pipe_lr = fit_pipeline(pipe_lr, X_train, y_train)
    print("    Both models successfully fitted.")

    # ── 3. Feature Mapping Verification ─────────────────────────────
    print("\n[3] Verifying feature mapping after ColumnTransformer...")
    feat_names = get_preprocessed_feature_names(pipe_xgb, X_train)
    print(f"    Extracted {len(feat_names)} preprocessed features:")
    print(f"    First 5: {feat_names[:5]}")
    print(f"    Last 5 : {feat_names[-5:]}")

    # ── 4. SHAP Computation & Reconstruction Verification ────────────
    print("\n[4] Computing SHAP values & verifying mathematical reconstruction...")
    shap_vals, base_val, names, X_trans = compute_shap_explanation(pipe_xgb, X_test)
    print(f"    SHAP values matrix shape: {shap_vals.shape}")
    print(f"    Base value (logit): {base_val:.4f}")

    recon_results = verify_shap_reconstruction(pipe_xgb, X_test)
    print(f"    Reconstruction Validated : {recon_results['is_valid']}")
    print(f"    Max Logit Margin Error   : {recon_results['max_margin_error']}")
    print(f"    Max Probability Error    : {recon_results['max_probability_error']}")

    assert recon_results["is_valid"], f"SHAP reconstruction failed! Max diff: {recon_results['max_margin_error']}"

    # ── 5. Generate Global Explainability Visualizations ────────────
    print("\n[5] Generating Global Explainability Artifacts...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shap_img_path = OUTPUT_DIR / "shap_feature_importance.png"
    plot_global_shap_importance(shap_vals, names, shap_img_path, top_n=15)

    xgb_img_path = OUTPUT_DIR / "xgboost_feature_importance.png"
    plot_xgboost_importance_comparison(pipe_xgb, X_train, xgb_img_path, top_n=15)

    summary_img_path = OUTPUT_DIR / "shap_summary.png"
    plot_shap_summary(pipe_xgb, X_test, summary_img_path, max_display=15)

    # ── 6. Individual Applicant Explanation ──────────────────────────
    print("\n[6] Generating Individual Loan Application Explanation...")
    # Find a sample high-risk defaulted applicant (BAD=1) in X_test
    test_defaults = X_test[y_test == 1]
    sample_record = test_defaults.iloc[0]

    explanation = explain_loan_application(pipe_xgb, sample_record, feature_columns, top_n=5)

    print("\n" + "─" * 60)
    print("  SAMPLE INDIVIDUAL LOAN EXPLANATION (High-Risk Applicant)")
    print("─" * 60)
    print(f"  Predicted Default Probability : {explanation['predicted_default_pct']}")
    print(f"  Base Portfolio Risk          : {explanation['base_portfolio_risk_pct']}")
    print(f"  Underwriting Decision        : {explanation['decision']}")
    print("\n  Top Risk Drivers (Increasing Default Risk):")
    for f in explanation["top_positive_risk_factors"]:
        print(f"    + {f['description']:<35} (SHAP impact: {f['shap_impact']:>+6.4f})")
    print("\n  Top Risk Mitigants (Decreasing Default Risk):")
    for f in explanation["top_negative_risk_factors"]:
        print(f"    - {f['description']:<35} (SHAP impact: {f['shap_impact']:>+6.4f})")
    print("─" * 60)

    # Save JSON artifact
    json_path = OUTPUT_DIR / "sample_loan_explanation.json"
    json_path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")
    print(f"  Saved sample loan explanation → {json_path}")

    # ── 7. Logistic Regression Interpretability ─────────────────────
    print("\n[7] Extracting Logistic Regression Coefficients...")
    df_lr_coef = extract_logistic_coefficients(pipe_lr, X_train)

    print("\n" + "─" * 60)
    print("  LOGISTIC REGRESSION COEFFICIENT RANKING")
    print("─" * 60)
    print(f"  {'Feature':<25}  {'Coef (β)':>9}  {'Odds Ratio':>10}  {'Direction':<25}")
    print("  " + "─" * 72)
    for row in df_lr_coef.head(10).itertuples():
        print(f"  {row.feature:<25}  {row.coefficient:>9.4f}  {row.odds_ratio:>10.4f}  {row.odds_direction:<25}")
    print("─" * 60)

    # Save LR coefficients CSV
    lr_csv_path = OUTPUT_DIR / "logistic_regression_coefficients.csv"
    df_lr_coef.to_csv(lr_csv_path, index=False)
    print(f"  Saved Logistic Regression coefficients → {lr_csv_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 3 VERIFICATION COMPLETE — ALL TESTS PASSED")
    print(f"  Artifacts saved to: {OUTPUT_DIR}/")
    print("    - shap_feature_importance.png")
    print("    - xgboost_feature_importance.png")
    print("    - shap_summary.png")
    print("    - sample_loan_explanation.json")
    print("    - logistic_regression_coefficients.csv")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
