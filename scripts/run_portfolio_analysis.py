"""
Phase 4: Portfolio Risk & Expected Credit Loss (ECL) Analysis Script.

Usage
-----
  python scripts/run_portfolio_analysis.py
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
from src.ecl import (
    DEFAULT_LGD,
    calculate_ecl_by_category,
    calculate_ecl_by_risk_segment,
    calculate_portfolio_ecl,
    compute_portfolio_kpis,
    validate_risk_segments,
)
from src.features import engineer_features
from src.modeling import build_pipeline, fit_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "hmeq.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "portfolio"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    print("\n" + "=" * 65)
    print("  CREDITRISK AI — PHASE 4: PORTFOLIO ECL & RISK ANALYTICS")
    print("=" * 65)

    # ── 1. Load Data ────────────────────────────────────────────────
    print("\n[1] Loading HMEQ dataset...")
    raw = load_hmeq(DEFAULT_DATASET)
    print(f"    Raw records: {len(raw)} rows")

    # ── 2. Feature Engineering & Preparation ────────────────────────
    print("[2] Engineering features & preparing portfolio frame...")
    enriched = engineer_features(raw)
    enriched = enriched.rename(columns={"bad": "target"})
    prepared = prepare_portfolio_frame(enriched, random_state=RANDOM_STATE)
    X, feature_columns = prepare_features(prepared)
    y = prepared["target"].astype(int)

    # ── 3. Train Model ──────────────────────────────────────────────
    print("[3] Fitting XGBoost model for portfolio scoring...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline(X_train, "xgboost", random_state=RANDOM_STATE)
    pipeline = fit_pipeline(pipeline, X_train, y_train)
    print("    Model fitted successfully.")

    # ── 4. Score Full Portfolio ──────────────────────────────────────
    print("\n[4] Scoring entire portfolio (N = 5,960)...")
    # Score the entire dataset X for portfolio risk reporting
    probs = pipeline.predict_proba(X)[:, 1]
    prepared["default_probability"] = probs

    # Restore raw 'job' and 'reason' for concentration reporting
    prepared["job"] = raw["job"].values
    prepared["reason"] = raw["reason"].values
    if "bad" not in prepared.columns:
        prepared["bad"] = raw["bad"].values

    # ── 5. Calculate ECL (PD × LGD × EAD) ─────────────────────────────
    print(f"[5] Applying ECL framework (PD × LGD={DEFAULT_LGD} × EAD=loan)...")
    scored_portfolio = calculate_portfolio_ecl(
        df=prepared,
        pd_col="default_probability",
        lgd_val=DEFAULT_LGD,
        ead_col="loan",
    )

    # ── 6. Portfolio Aggregations ────────────────────────────────────
    print("[6] Computing portfolio risk summaries...")
    kpis = compute_portfolio_kpis(scored_portfolio)
    segment_summary = calculate_ecl_by_risk_segment(scored_portfolio)
    job_summary = calculate_ecl_by_category(scored_portfolio, category_col="job")
    reason_summary = calculate_ecl_by_category(scored_portfolio, category_col="reason")

    # Validate risk segments monotonicity
    val_result = validate_risk_segments(segment_summary)
    print(f"    Risk Segment Ordering Validated : {val_result['is_valid']}")

    # ── 7. Display Results ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  PORTFOLIO EXECUTIVE RISK SUMMARY")
    print("─" * 65)
    print(f"  Total Portfolio Loans       : {kpis['portfolio_loan_count']:,}")
    print(f"  Total Portfolio Exposure    : ${kpis['total_exposure_ead']:,.2f}")
    print(f"  Total Expected Credit Loss  : ${kpis['total_expected_credit_loss_ecl']:,.2f}")
    print(f"  Portfolio Expected Loss Rate: {kpis['portfolio_loss_rate_pct']}")
    print(f"  Weighted Average PD         : {kpis['weighted_average_pd'] * 100:.2f}%")
    print(f"  Historical Observed BAD Rate: {kpis['observed_historical_bad_rate'] * 100:.2f}%")
    print(f"  High Risk Exposure (EAD)    : ${kpis['high_risk_exposure_ead']:,.2f} ({kpis['high_risk_exposure_pct']}%)")
    print(f"  High Risk ECL Contribution  : ${kpis['high_risk_ecl_contribution']:,.2f} ({kpis['high_risk_ecl_pct']}%)")
    print("─" * 65)

    print("\n  ECL BY RISK SEGMENT")
    print(f"  {'Segment':<12}  {'Loans':>6}  {'% Loans':>8}  {'Total EAD ($)':>14}  {'Avg PD':>8}  {'Total ECL ($)':>14}  {'% ECL':>8}")
    print("  " + "─" * 78)
    for r in segment_summary.itertuples():
        print(
            f"  {r.risk_segment:<12}  {r.loan_count:>6d}  {r.pct_of_loans:>7.1f}%  "
            f"${r.total_ead:>13,.2f}  {r.avg_pd*100:>7.1f}%  ${r.total_ecl:>13,.2f}  {r.pct_of_ecl:>7.1f}%"
        )
    print("─" * 78)

    print("\n  CONCENTRATION ANALYSIS — BY JOB CATEGORY")
    print(f"  {'Job':<12}  {'Loans':>6}  {'Total EAD ($)':>14}  {'Avg PD':>8}  {'Hist BAD%':>10}  {'Total ECL ($)':>14}  {'% Total ECL':>11}")
    print("  " + "─" * 84)
    for r in job_summary.itertuples():
        hist_str = f"{r.historical_default_rate*100:.1f}%" if isinstance(r.historical_default_rate, float) else "N/A"
        print(
            f"  {r.job:<12}  {r.loan_count:>6d}  ${r.total_ead:>13,.2f}  {r.avg_pd*100:>7.1f}%  "
            f"{hist_str:>10}  ${r.total_ecl:>13,.2f}  {r.pct_of_total_ecl:>10.1f}%"
        )
    print("─" * 84)

    # ── 8. Save Artifacts ────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "portfolio_summary.json"
    json_path.write_text(json.dumps(kpis, indent=2), encoding="utf-8")

    segment_summary.to_csv(OUTPUT_DIR / "ecl_by_risk_segment.csv", index=False)
    job_summary.to_csv(OUTPUT_DIR / "ecl_by_job.csv", index=False)
    reason_summary.to_csv(OUTPUT_DIR / "ecl_by_reason.csv", index=False)
    scored_portfolio.to_csv(OUTPUT_DIR / "scored_portfolio.csv", index=False)

    print(f"\n  Artifacts saved to: {OUTPUT_DIR}/")
    print("    - portfolio_summary.json")
    print("    - ecl_by_risk_segment.csv")
    print("    - ecl_by_job.csv")
    print("    - ecl_by_reason.csv")
    print("    - scored_portfolio.csv")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
