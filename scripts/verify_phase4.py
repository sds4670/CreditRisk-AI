"""
Phase 4 Verification Script — Portfolio Expected Credit Loss (ECL) & Risk Analytics.

Usage
-----
  python scripts/verify_phase4.py
"""
from __future__ import annotations

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
    RISK_SEGMENT_ORDER,
    calculate_ecl_by_category,
    calculate_ecl_by_risk_segment,
    calculate_portfolio_ecl,
    compute_portfolio_kpis,
    validate_risk_segments,
)
from src.features import engineer_features
from src.modeling import build_pipeline, fit_pipeline

DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "hmeq.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    print("\n" + "=" * 65)
    print("  CREDITRISK AI — PHASE 4: VERIFICATION SUITE")
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

    # ── 1. Setup & Scoring ──────────────────────────────────────────
    print("\n[1] Scoring portfolio with XGBoost model...")
    raw = load_hmeq(DEFAULT_DATASET)
    enriched = engineer_features(raw)
    enriched = enriched.rename(columns={"bad": "target"})
    prepared = prepare_portfolio_frame(enriched, random_state=RANDOM_STATE)
    X, feature_columns = prepare_features(prepared)
    y = prepared["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline(X_train, "xgboost", random_state=RANDOM_STATE)
    pipeline = fit_pipeline(pipeline, X_train, y_train)

    probs = pipeline.predict_proba(X)[:, 1]
    prepared["default_probability"] = probs
    prepared["job"] = raw["job"].values
    prepared["reason"] = raw["reason"].values

    scored = calculate_portfolio_ecl(
        prepared, pd_col="default_probability", lgd_val=DEFAULT_LGD, ead_col="loan"
    )

    # ── 2. Boundary & Domain Checks ─────────────────────────────────
    print("\n[2] Verifying mathematical boundaries & domains...")

    # PD in [0, 1]
    pds = scored["default_probability"]
    pd_valid = bool((pds >= 0.0).all() and (pds <= 1.0).all())
    check("PD values strictly in [0, 1]", pd_valid, f"Min PD: {pds.min():.4f}, Max PD: {pds.max():.4f}")

    # LGD in [0, 1]
    lgds = scored["lgd"]
    lgd_valid = bool((lgds >= 0.0).all() and (lgds <= 1.0).all())
    check("LGD values strictly in [0, 1]", lgd_valid, f"LGD value: {lgds.iloc[0]}")

    # EAD >= 0
    eads = scored["ead"]
    ead_valid = bool((eads >= 0.0).all())
    check("EAD values non-negative (EAD >= 0)", ead_valid, f"Min EAD: ${eads.min():,.2f}")

    # ECL >= 0
    ecls = scored["ecl"]
    ecl_valid = bool((ecls >= 0.0).all())
    check("ECL values non-negative (ECL >= 0)", ecl_valid, f"Min ECL: ${ecls.min():,.2f}")

    # ── 3. Mathematical Formula Equality ───────────────────────────
    print("\n[3] Verifying mathematical identity: ECL == PD × LGD × EAD...")
    calculated_ecl = scored["default_probability"] * scored["lgd"] * scored["ead"]
    max_ecl_diff = float(np.max(np.abs(scored["ecl"] - calculated_ecl)))
    check("ECL == PD × LGD × EAD for all rows", max_ecl_diff < 1e-5, f"Max absolute difference: {max_ecl_diff:.2e}")

    # Total ECL sum equality
    sum_individual_ecl = float(scored["ecl"].sum())
    kpis = compute_portfolio_kpis(scored)
    kpi_ecl = kpis["total_expected_credit_loss_ecl"]
    ecl_sum_diff = abs(sum_individual_ecl - kpi_ecl)
    check("Total ECL == sum of individual ECLs", ecl_sum_diff < 0.01, f"Sum: ${sum_individual_ecl:,.2f} vs KPI: ${kpi_ecl:,.2f}")

    # ── 4. Risk Segmentation & Monotonicity ──────────────────────────
    print("\n[4] Verifying risk segmentation labels & monotonicity...")
    valid_labels = set(RISK_SEGMENT_ORDER)
    observed_labels = set(scored["risk_segment"].unique())
    check("Risk segments contain valid labels", observed_labels.issubset(valid_labels), f"Observed labels: {observed_labels}")

    segment_summary = calculate_ecl_by_risk_segment(scored)
    val_mon = validate_risk_segments(segment_summary)
    check("Average PD increases monotonically across risk segments", val_mon["pd_monotonic_increasing"], f"Segment PDs: {val_mon['segment_pds']}")
    check("Average ECL increases monotonically across risk segments", val_mon["ecl_monotonic_increasing"], f"Segment ECLs: {val_mon['segment_ecls']}")

    # ── 5. Internal Consistency & Target Leakage ────────────────────
    print("\n[5] Verifying portfolio totals & leak-free integrity...")

    # Loan count consistency
    total_loans = len(scored)
    sum_segment_loans = int(segment_summary["loan_count"].sum())
    check("Portfolio loan count is consistent", total_loans == sum_segment_loans, f"Scored: {total_loans} vs Segments: {sum_segment_loans}")

    # EAD sum consistency
    sum_segment_ead = float(segment_summary["total_ead"].sum())
    total_ead = float(scored["ead"].sum())
    check("Portfolio total EAD is consistent", abs(total_ead - sum_segment_ead) < 1.0, f"Scored: ${total_ead:,.2f} vs Segments: ${sum_segment_ead:,.2f}")

    # No leakage into features
    check("Target columns ('bad', 'target') absent from feature matrix", "bad" not in feature_columns and "target" not in feature_columns)

    # Model prediction stability (probs unchanged by ECL calculation)
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    scored_test_probs = scored.loc[X_test.index, "default_probability"].values
    prob_diff = float(np.max(np.abs(test_probs - scored_test_probs)))
    check("Original model default probabilities preserved untouched", prob_diff < 1e-6, f"Max difference: {prob_diff:.2e}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"  TOTAL CHECKS: {passed} passed  |  {failed} failed")
    print("=" * 65 + "\n")

    assert failed == 0, f"Verification failed with {failed} errors!"


if __name__ == "__main__":
    main()
