"""
Expected Credit Loss (ECL) and Portfolio Risk Analytics — Phase 4.

This module provides portfolio-level risk analytics implementing the standard
financial risk framework:

    ECL = PD × LGD × EAD

Conceptual Framework & Assumptions
-----------------------------------
1. Probability of Default (PD):
   Extracted directly from the predicted probability of the trained XGBoost model
   (``default_probability`` column).

2. Loss Given Default (LGD):
   Configurable assumption (default = 0.45 or 45%).  HMEQ does not contain real
   recovery or collateral disposition data.  LGD is an illustrative parameter.

3. Exposure at Default (EAD):
   Simplified proxy using loan request amount (``loan`` column).  HMEQ does not
   provide actual outstanding balance or credit line draw information at default.

4. Expected Credit Loss (ECL):
   Calculated element-wise as ``pd * lgd * ead``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Default LGD assumption (45% for unsecured/partially secured retail loans)
DEFAULT_LGD: float = 0.45

# Standard Risk Segment Cutoffs based on Probability of Default (PD)
# Low: PD < 10% | Moderate: 10% <= PD < 25% | High: 25% <= PD < 50% | Very High: PD >= 50%
RISK_SEGMENT_CUTOFFS: tuple[tuple[float, str], ...] = (
    (0.10, "Low"),
    (0.25, "Moderate"),
    (0.50, "High"),
    (1.01, "Very High"),
)

RISK_SEGMENT_ORDER: tuple[str, ...] = ("Low", "Moderate", "High", "Very High")


# ---------------------------------------------------------------------------
# Core Calculation Functions
# ---------------------------------------------------------------------------


def calculate_ecl(
    pd: float | np.ndarray | pd.Series,
    lgd: float | np.ndarray | pd.Series = DEFAULT_LGD,
    ead: float | np.ndarray | pd.Series = 0.0,
) -> float | np.ndarray | pd.Series:
    """Calculate Expected Credit Loss using element-wise vector operations.

    Formula:
        ECL = PD × LGD × EAD

    Parameters
    ----------
    pd:
        Probability of Default in [0, 1].
    lgd:
        Loss Given Default in [0, 1] (default = 0.45).
    ead:
        Exposure at Default ($) >= 0.

    Returns
    -------
    float | np.ndarray | pd.Series
        Expected Credit Loss in currency units ($).
    """
    return pd * lgd * ead


def assign_risk_segments(
    df: pd.DataFrame, pd_col: str = "default_probability"
) -> pd.Series:
    """Categorise loan applications into risk segments based on predicted PD.

    Segmentation buckets:
    - Low Risk       : PD < 10%
    - Moderate Risk  : 10% <= PD < 25%
    - High Risk      : 25% <= PD < 50%
    - Very High Risk : PD >= 50%

    Parameters
    ----------
    df:
        DataFrame containing the predicted probability column.
    pd_col:
        Column name for predicted default probability.

    Returns
    -------
    pd.Series
        Categorical Series with values in ('Low', 'Moderate', 'High', 'Very High').
    """
    pd_series = pd.to_numeric(df[pd_col], errors="coerce").fillna(0.0)

    conditions = [
        pd_series < 0.10,
        (pd_series >= 0.10) & (pd_series < 0.25),
        (pd_series >= 0.25) & (pd_series < 0.50),
        pd_series >= 0.50,
    ]
    choices = ["Low", "Moderate", "High", "Very High"]

    segmented = pd.Series(
        np.select(conditions, choices, default="Very High"),
        index=df.index,
        name="risk_segment",
    )
    return pd.Categorical(segmented, categories=RISK_SEGMENT_ORDER, ordered=True)


def calculate_portfolio_ecl(
    df: pd.DataFrame,
    pd_col: str = "default_probability",
    lgd_val: float = DEFAULT_LGD,
    ead_col: str = "loan",
) -> pd.DataFrame:
    """Enrich scored portfolio DataFrame with LGD, EAD, ECL, and Risk Segment.

    Parameters
    ----------
    df:
        Scored portfolio DataFrame containing PD and loan columns.
    pd_col:
        Column name for predicted default probability.
    lgd_val:
        Configurable LGD assumption value (default = 0.45).
    ead_col:
        Column name for Exposure at Default proxy (default = 'loan').

    Returns
    -------
    pd.DataFrame
        New DataFrame enriched with:
        default_probability, lgd, ead, ecl, risk_segment.
        Original columns are preserved untouched.
    """
    output = df.copy()

    if pd_col not in output.columns:
        raise ValueError(f"Required PD column '{pd_col}' not found in DataFrame.")

    # EAD fallback
    if ead_col not in output.columns:
        if "ead" in output.columns:
            ead_col = "ead"
        else:
            raise ValueError(f"Required EAD column '{ead_col}' not found in DataFrame.")

    output["default_probability"] = pd.to_numeric(output[pd_col], errors="coerce").clip(0.0, 1.0)
    output["lgd"] = float(lgd_val)
    output["ead"] = pd.to_numeric(output[ead_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    output["ecl"] = calculate_ecl(
        pd=output["default_probability"],
        lgd=output["lgd"],
        ead=output["ead"],
    )
    output["risk_segment"] = assign_risk_segments(output, pd_col="default_probability")

    return output


# ---------------------------------------------------------------------------
# Portfolio Analytics & Aggregations
# ---------------------------------------------------------------------------


def calculate_weighted_average_pd(
    df: pd.DataFrame,
    pd_col: str = "default_probability",
    ead_col: str = "ead",
) -> float:
    """Calculate Exposure-Weighted Average Probability of Default.

    Formula:
        Weighted PD = Σ(PD_i × EAD_i) / Σ(EAD_i)
    """
    pds = pd.to_numeric(df[pd_col], errors="coerce").fillna(0.0)
    eads = pd.to_numeric(df[ead_col], errors="coerce").fillna(0.0)

    total_ead = eads.sum()
    if total_ead <= 0:
        return float(pds.mean()) if len(pds) > 0 else 0.0

    weighted_pd = float((pds * eads).sum() / total_ead)
    return round(weighted_pd, 6)


def calculate_portfolio_ead(df: pd.DataFrame, ead_col: str = "ead") -> float:
    """Calculate total exposure at default across portfolio."""
    return float(pd.to_numeric(df[ead_col], errors="coerce").fillna(0.0).sum())


def calculate_ecl_by_risk_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ECL metrics grouped by Risk Segment (Low, Moderate, High, Very High).

    Returns
    -------
    pd.DataFrame
        Columns: risk_segment, loan_count, pct_of_loans, total_ead,
        pct_of_ead, avg_pd, total_ecl, avg_ecl, pct_of_ecl.
    """
    required = {"risk_segment", "default_probability", "ead", "ecl"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame missing required ECL columns: {required - set(df.columns)}")

    total_portfolio_count = len(df)
    total_portfolio_ead = df["ead"].sum()
    total_portfolio_ecl = df["ecl"].sum()

    grouped = df.groupby("risk_segment", observed=False)

    records = []
    for segment in RISK_SEGMENT_ORDER:
        if segment in grouped.groups:
            group = grouped.get_group(segment)
            count = len(group)
            ead = float(group["ead"].sum())
            ecl = float(group["ecl"].sum())
            avg_pd = calculate_weighted_average_pd(group, "default_probability", "ead")
            avg_ecl = float(ecl / count) if count > 0 else 0.0
        else:
            count = 0
            ead = 0.0
            ecl = 0.0
            avg_pd = 0.0
            avg_ecl = 0.0

        records.append({
            "risk_segment": segment,
            "loan_count": count,
            "pct_of_loans": round(count / total_portfolio_count * 100, 2) if total_portfolio_count else 0.0,
            "total_ead": round(ead, 2),
            "pct_of_ead": round(ead / total_portfolio_ead * 100, 2) if total_portfolio_ead else 0.0,
            "avg_pd": round(avg_pd, 4),
            "total_ecl": round(ecl, 2),
            "avg_ecl": round(avg_ecl, 2),
            "pct_of_ecl": round(ecl / total_portfolio_ecl * 100, 2) if total_portfolio_ecl else 0.0,
        })

    return pd.DataFrame(records)


def calculate_ecl_by_category(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Aggregate ECL metrics grouped by a categorical portfolio variable (e.g. 'job', 'reason').

    Returns
    -------
    pd.DataFrame
        Columns: category, loan_count, total_ead, avg_pd,
        historical_default_rate, total_ecl, pct_of_total_ecl.
    """
    if category_col not in df.columns:
        raise ValueError(f"Category column '{category_col}' not found in DataFrame.")

    total_portfolio_ecl = df["ecl"].sum()
    # Check if historical target column exists (bad or target)
    target_col = "bad" if "bad" in df.columns else ("target" if "target" in df.columns else None)

    records = []
    # Fill NA categories for aggregation
    temp_df = df.copy()
    temp_df[category_col] = temp_df[category_col].fillna("Unknown").astype(str)

    for cat_val, group in temp_df.groupby(category_col):
        count = len(group)
        ead = float(group["ead"].sum())
        ecl = float(group["ecl"].sum())
        avg_pd = calculate_weighted_average_pd(group, "default_probability", "ead")

        hist_bad_rate = (
            float(group[target_col].mean())
            if target_col is not None and group[target_col].notna().any()
            else np.nan
        )

        records.append({
            category_col: cat_val,
            "loan_count": count,
            "total_ead": round(ead, 2),
            "avg_pd": round(avg_pd, 4),
            "historical_default_rate": round(hist_bad_rate, 4) if not np.isnan(hist_bad_rate) else "N/A",
            "total_ecl": round(ecl, 2),
            "pct_of_total_ecl": round(ecl / total_portfolio_ecl * 100, 2) if total_portfolio_ecl else 0.0,
        })

    result = pd.DataFrame(records).sort_values(by="total_ecl", ascending=False).reset_index(drop=True)
    return result


def calculate_portfolio_concentration(
    df: pd.DataFrame, category_cols: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Compute concentration tables across key portfolio dimensions."""
    if category_cols is None:
        category_cols = [c for c in ["job", "reason"] if c in df.columns]

    concentration = {}
    for col in category_cols:
        concentration[col] = calculate_ecl_by_category(df, category_col=col)

    return concentration


# ---------------------------------------------------------------------------
# Portfolio KPIs & Validation
# ---------------------------------------------------------------------------


def compute_portfolio_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute top-level executive portfolio risk KPIs.

    Distinguishes predicted model risk metrics from historical observed BAD rate.
    """
    total_count = int(len(df))
    total_ead = float(df["ead"].sum())
    total_ecl = float(df["ecl"].sum())

    weighted_pd = calculate_weighted_average_pd(df, "default_probability", "ead")
    unweighted_pd = float(df["default_probability"].mean())
    avg_lgd = float(df["lgd"].mean()) if "lgd" in df.columns else DEFAULT_LGD

    ecl_loss_rate = float(total_ecl / total_ead) if total_ead > 0 else 0.0

    # High risk exposure (PD >= 25%, High + Very High segments)
    high_risk_mask = df["default_probability"] >= 0.25
    high_risk_ead = float(df.loc[high_risk_mask, "ead"].sum())
    high_risk_ecl = float(df.loc[high_risk_mask, "ecl"].sum())

    pct_high_risk_ead = float(high_risk_ead / total_ead * 100) if total_ead > 0 else 0.0
    pct_high_risk_ecl = float(high_risk_ecl / total_ecl * 100) if total_ecl > 0 else 0.0

    # Historical observed BAD rate if available
    target_col = "bad" if "bad" in df.columns else ("target" if "target" in df.columns else None)
    observed_bad_rate = float(df[target_col].mean()) if target_col is not None else None

    return {
        "portfolio_loan_count": total_count,
        "total_exposure_ead": round(total_ead, 2),
        "total_expected_credit_loss_ecl": round(total_ecl, 2),
        "weighted_average_pd": round(weighted_pd, 4),
        "unweighted_average_pd": round(unweighted_pd, 4),
        "assumed_lgd": round(avg_lgd, 4),
        "portfolio_loss_rate_ecl_over_ead": round(ecl_loss_rate, 4),
        "portfolio_loss_rate_pct": f"{ecl_loss_rate * 100:.2f}%",
        "high_risk_exposure_ead": round(high_risk_ead, 2),
        "high_risk_exposure_pct": round(pct_high_risk_ead, 2),
        "high_risk_ecl_contribution": round(high_risk_ecl, 2),
        "high_risk_ecl_pct": round(pct_high_risk_ecl, 2),
        "observed_historical_bad_rate": round(observed_bad_rate, 4) if observed_bad_rate is not None else "N/A",
        "assumptions_disclaimer": (
            "LOAN is used as simplified EAD proxy; LGD=0.45 is an illustrative assumption; "
            "not a regulatory-grade IFRS 9 ECL implementation."
        ),
    }


def validate_risk_segments(segment_summary: pd.DataFrame) -> dict[str, Any]:
    """Verify logical ordering of risk segments (Average PD & ECL increase Low -> Very High).

    Parameters
    ----------
    segment_summary:
        DataFrame output of ``calculate_ecl_by_risk_segment()``.

    Returns
    -------
    dict
        Validation status and check details.
    """
    df_sorted = segment_summary.copy()

    pds = df_sorted["avg_pd"].values
    ecls = df_sorted["avg_ecl"].values

    # Check monotonicity
    pd_monotonic = bool(np.all(np.diff(pds) >= -1e-6))
    ecl_monotonic = bool(np.all(np.diff(ecls) >= -1e-6))

    is_valid = pd_monotonic and ecl_monotonic

    return {
        "is_valid": is_valid,
        "pd_monotonic_increasing": pd_monotonic,
        "ecl_monotonic_increasing": ecl_monotonic,
        "segment_pds": [float(x) for x in pds],
        "segment_ecls": [float(x) for x in ecls],
    }
