from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import prepare_features, prepare_portfolio_frame
from src.ecl import calculate_portfolio_ecl, DEFAULT_LGD
from src.features import engineer_features

RISK_BINS = [-0.001, 0.10, 0.25, 0.50, 1.00]
RISK_LABELS = ["Low", "Moderate", "High", "Very High"]


def score_portfolio(
    raw_df: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
    lgd_val: float = DEFAULT_LGD,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Score raw portfolio DataFrame using trained pipeline and calculate ECL."""
    # Ensure column names are lowercased
    clean_df = raw_df.copy()
    clean_df.columns = [str(c).strip().lower() for c in clean_df.columns]

    # Run feature engineering
    enriched = engineer_features(clean_df)

    # Run portfolio frame preparation
    portfolio = prepare_portfolio_frame(enriched)
    X, _ = prepare_features(portfolio, feature_columns=feature_columns)

    probabilities = pipeline.predict_proba(X)[:, 1]
    scored = portfolio.copy()
    scored["default_probability"] = np.round(probabilities, 4)
    scored["predicted_default"] = (scored["default_probability"] >= threshold).astype(int)

    # Attach raw job and reason if available
    if "job" in clean_df.columns:
        scored["job"] = clean_df["job"].values
    if "reason" in clean_df.columns:
        scored["reason"] = clean_df["reason"].values

    # Determine loan / EAD column
    ead_col = "loan" if "loan" in scored.columns else ("loan_amnt" if "loan_amnt" in scored.columns else None)
    if ead_col is not None:
        scored["loan"] = pd.to_numeric(scored[ead_col], errors="coerce").fillna(0.0)

    scored = calculate_portfolio_ecl(
        scored, pd_col="default_probability", lgd_val=lgd_val, ead_col="loan" if "loan" in scored.columns else "loan_amnt"
    )

    scored["expected_loss"] = scored["ecl"]
    return scored



def build_delinquency_trend(scored: pd.DataFrame) -> pd.DataFrame:
    trend_source = scored.copy()
    trend_source["issue_date"] = pd.to_datetime(trend_source["issue_date"], errors="coerce")
    trend_source = trend_source.dropna(subset=["issue_date"]).copy()
    trend_source["month"] = trend_source["issue_date"].dt.to_period("M").dt.to_timestamp()

    aggregations = {
        "default_probability": "mean",
        "days_past_due": "mean",
        "loan_amnt": "sum",
        "predicted_default": "mean",
    }
    if "target" in trend_source.columns:
        aggregations["target"] = "mean"

    trend = trend_source.groupby("month", as_index=False).agg(aggregations)
    trend = trend.rename(
        columns={
            "default_probability": "avg_default_probability",
            "days_past_due": "avg_days_past_due",
            "loan_amnt": "portfolio_exposure",
            "predicted_default": "predicted_default_rate",
            "target": "observed_default_rate",
        }
    )
    return trend.sort_values("month")


def build_risk_segmentation(scored: pd.DataFrame) -> pd.DataFrame:
    loan_amount = pd.to_numeric(scored.get("loan_amnt", 0), errors="coerce").fillna(0.0)
    segment_source = scored.copy()
    segment_source["loan_amount_clean"] = loan_amount

    segmentation = (
        segment_source.groupby("risk_segment", as_index=False)
        .agg(
            loan_count=("risk_segment", "size"),
            exposure=("loan_amount_clean", "sum"),
            avg_pd=("default_probability", "mean"),
            expected_loss=("expected_loss", "sum"),
        )
        .sort_values("avg_pd")
    )
    return segmentation
