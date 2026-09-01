"""
Phase 5 Verification Script — Professional Dashboard Integration.

Usage
-----
  python scripts/verify_phase5.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    print("\n" + "=" * 65)
    print("  CREDITRISK AI — PHASE 5: VERIFICATION SUITE")
    print("=" * 65)

    PASS = "[PASS]"
    FAIL = "[FAIL]"
    results = []

    def check(label: str, condition: bool, detail: str = ""):
        status = PASS if condition else FAIL
        msg = f"  {status}  {label}"
        if detail:
            msg += f"\n         {detail}"
        print(msg)
        results.append((label, condition))

    # ── 1. Backend Modules Import ────────────────────────────────────
    print("\n[1] Verifying backend module imports...")
    backend_modules = [
        "src.dataset_loader",
        "src.features",
        "src.modeling",
        "src.evaluation",
        "src.explainability",
        "src.ecl",
        "src.scoring",
    ]
    for mod_name in backend_modules:
        try:
            importlib.import_module(mod_name)
            check(f"Module '{mod_name}' imported successfully", True)
        except Exception as exc:
            check(f"Module '{mod_name}' imported successfully", False, str(exc))

    # ── 2. Dashboard Dependencies ───────────────────────────────────
    print("\n[2] Verifying Streamlit dashboard dependencies...")
    deps = ["streamlit", "plotly", "shap", "xgboost", "matplotlib"]
    for dep in deps:
        try:
            importlib.import_module(dep)
            check(f"Dependency '{dep}' available", True)
        except Exception as exc:
            check(f"Dependency '{dep}' available", False, str(exc))

    # ── 3. Model & Comparison Artifacts ──────────────────────────────
    print("\n[3] Verifying required model & comparison artifacts...")
    artifacts_dir = PROJECT_ROOT / "artifacts"
    check("artifacts/model.joblib exists", (artifacts_dir / "model.joblib").exists())
    check("artifacts/metadata.json exists", (artifacts_dir / "metadata.json").exists())
    check("artifacts/model_comparison.json exists", (artifacts_dir / "model_comparison.json").exists())

    # ── 4. Analytics & Visualization Artifacts ───────────────────────
    print("\n[4] Verifying explainability & portfolio artifacts...")
    explain_dir = artifacts_dir / "explainability"
    portfolio_dir = artifacts_dir / "portfolio"

    check("artifacts/explainability/shap_feature_importance.png exists", (explain_dir / "shap_feature_importance.png").exists())
    check("artifacts/explainability/xgboost_feature_importance.png exists", (explain_dir / "xgboost_feature_importance.png").exists())
    check("artifacts/explainability/shap_summary.png exists", (explain_dir / "shap_summary.png").exists())
    check("artifacts/explainability/sample_loan_explanation.json exists", (explain_dir / "sample_loan_explanation.json").exists())

    check("artifacts/portfolio/portfolio_summary.json exists", (portfolio_dir / "portfolio_summary.json").exists())
    check("artifacts/portfolio/scored_portfolio.csv exists", (portfolio_dir / "scored_portfolio.csv").exists())

    # ── 5. ECL Formula Correctness ──────────────────────────────────
    print("\n[5] Verifying ECL formula calculation correctness...")
    from src.ecl import calculate_ecl
    pd_val, lgd_val, ead_val = 0.20, 0.45, 10000.0
    expected_ecl = 0.20 * 0.45 * 10000.0  # = 900.0
    calc_ecl = float(calculate_ecl(pd_val, lgd_val, ead_val))
    check("ECL formula: 0.20 × 0.45 × 10000 = 900.0", abs(calc_ecl - expected_ecl) < 1e-5, f"Calculated: {calc_ecl}")

    # ── 6. Single Loan Scoring & Feature Engineering ────────────────
    print("\n[6] Verifying single loan scoring & feature generation...")
    from src.modeling import load_artifacts
    from src.scoring import score_portfolio

    pipeline, metadata = load_artifacts(artifacts_dir / "model.joblib", artifacts_dir / "metadata.json")
    feature_cols = metadata["feature_columns"]

    sample_loan = pd.DataFrame([{
        "loan": 15000, "mortdue": 50000, "value": 85000, "reason": "DebtCon",
        "job": "Office", "yoj": 6, "derog": 0, "delinq": 0, "clage": 180,
        "ninq": 1, "clno": 20, "debtinc": 32,
    }])

    try:
        scored = score_portfolio(sample_loan, pipeline=pipeline, feature_columns=feature_cols)
        check("Single loan scored successfully", len(scored) == 1)
        check("default_probability calculated", "default_probability" in scored.columns)
        check("risk_segment calculated", "risk_segment" in scored.columns)
        check("ecl calculated", "ecl" in scored.columns)
    except Exception as exc:
        check("Single loan scored successfully", False, str(exc))

    # ── 7. Target Leakage Prevention ────────────────────────────────
    print("\n[7] Verifying target leakage prevention...")
    forbidden = {"bad", "target", "days_past_due", "loan_amnt", "issue_date"}
    leaky_cols = [c for c in feature_cols if c in forbidden]
    check("No target/synthetic columns present in feature matrix", len(leaky_cols) == 0, f"Leaky columns found: {leaky_cols}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"  TOTAL CHECKS: {passed} passed  |  {failed} failed")
    print("=" * 65 + "\n")

    assert failed == 0, f"Phase 5 verification failed with {failed} errors!"


if __name__ == "__main__":
    main()
