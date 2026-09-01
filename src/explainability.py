r"""
Model Explainability and Interpretability for CreditRisk AI — Phase 3.

This module provides model-agnostic and model-specific explainability tools
for both tree-based models (XGBoost) and linear baselines (Logistic Regression).

Key Capabilities
----------------
1. Feature Name Mapping:
   Extracts preprocessed feature names from Scikit-Learn ColumnTransformers,
   cleaning prefixes (e.g. ``num__debtinc`` → ``debtinc``, ``cat__job_Office`` → ``job: Office``).

2. SHAP Value Generation (XGBoost):
   Computes exact TreeSHAP values for XGBoost models. Validates that
   ``base_value + sum(shap_values) == margin_prediction``.

3. Individual Applicant Explanation:
   Decomposes single loan decisions into top positive risk drivers (increasing default risk)
   and negative risk drivers (reducing default risk) with natural language summaries.

4. Logistic Regression Interpretability:
   Extracts fitted log-odds coefficients ($\beta$), odds ratios ($\exp(\beta)$),
   and ranks features by impact magnitude.

5. Visualizations:
   Generates publication-quality SHAP feature importance charts, XGBoost gain charts,
   and SHAP summary beeswarm plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy.special import expit  # Sigmoid function for logit -> probability conversion
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Feature Name Extraction & Preprocessing Helpers
# ---------------------------------------------------------------------------


def clean_feature_name(name: str) -> str:
    """Clean sklearn ColumnTransformer output names for display.

    Examples:
        'num__debtinc' -> 'debtinc'
        'cat__job_Office' -> 'job: Office'
        'cat__reason_DebtCon' -> 'reason: DebtCon'
    """
    if name.startswith("num__"):
        return name[5:]
    if name.startswith("cat__"):
        raw = name[5:]
        if "_" in raw:
            parts = raw.split("_", 1)
            return f"{parts[0]}: {parts[1]}"
        return raw
    return name


def get_preprocessed_feature_names(pipeline: Pipeline, X_sample: pd.DataFrame) -> list[str]:
    """Extract raw and cleaned feature names after ColumnTransformer preprocessing.

    Parameters
    ----------
    pipeline:
        Fitted or unfitted Pipeline containing a 'preprocessor' step.
    X_sample:
        Sample DataFrame matching the feature structure.

    Returns
    -------
    list[str]
        Cleaned feature names matching the transformed array columns.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        raw_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback if get_feature_names_out is unavailable
        num_cols = X_sample.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat_cols = [c for c in X_sample.columns if c not in num_cols]
        raw_names = [f"num__{c}" for c in num_cols] + [f"cat__{c}" for c in cat_cols]

    return [clean_feature_name(name) for name in raw_names]


def transform_features(pipeline: Pipeline, X_df: pd.DataFrame) -> np.ndarray:
    """Transform input DataFrame using the pipeline's preprocessor step.

    Returns dense 2D numpy float array.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=float)


# ---------------------------------------------------------------------------
# SHAP Computation & Verification
# ---------------------------------------------------------------------------


def compute_shap_explanation(
    pipeline: Pipeline, X_df: pd.DataFrame
) -> tuple[np.ndarray, float, list[str], np.ndarray]:
    """Compute TreeSHAP values for an XGBoost pipeline.

    Parameters
    ----------
    pipeline:
        Fitted Pipeline with 'preprocessor' and 'model' (XGBClassifier).
    X_df:
        Feature DataFrame (raw or pre-split).

    Returns
    -------
    tuple:
        (shap_values, base_value, feature_names, X_transformed)
        - shap_values: shape (N, n_features) in margin (logit) space.
        - base_value: float expected value $E[f(x)]$.
        - feature_names: list of cleaned feature names.
        - X_transformed: preprocessed feature matrix shape (N, n_features).
    """
    model = pipeline.named_steps["model"]
    feature_names = get_preprocessed_feature_names(pipeline, X_df)
    X_trans = transform_features(pipeline, X_df)

    if hasattr(model, "get_booster") or "XGB" in type(model).__name__:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_trans)
        if hasattr(explanation, "values"):
            shap_vals = explanation.values
            base_val = float(explanation.base_values[0]) if hasattr(explanation.base_values, "__len__") else float(explanation.base_values)
        else:
            shap_vals = explainer.shap_values(X_trans)
            base_val = float(explainer.expected_value)
    else:
        # Linear model fallback (Logistic Regression)
        coef = model.coef_[0]
        intercept = float(model.intercept_[0])
        # Linear SHAP: (x_j - mean_j) * coef_j
        mean_X = np.mean(X_trans, axis=0) if len(X_trans) > 1 else X_trans[0]
        base_val = intercept + float(np.dot(mean_X, coef))
        shap_vals = (X_trans - mean_X) * coef

    # Handle multi-output / binary array shape if needed
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # positive class (BAD=1)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    return shap_vals, base_val, feature_names, X_trans



def verify_shap_reconstruction(
    pipeline: Pipeline, X_df: pd.DataFrame, atol: float = 1e-3
) -> dict[str, Any]:
    """Verify that base_value + sum(shap_values) reconstructs model margin prediction.

    Mathematical guarantee:
        In margin (logit) space: base_value + sum_j(shap_values_ij) = margin_i
        And sigmoid(margin_i) = predict_proba(X_i)[1].

    Parameters
    ----------
    pipeline:
        Fitted XGBoost pipeline.
    X_df:
        Sample DataFrame to test.
    atol:
        Absolute tolerance for mathematical floating point equality.

    Returns
    -------
    dict
        Verification status, max reconstruction error, and sample count.
    """
    model = pipeline.named_steps["model"]
    X_trans = transform_features(pipeline, X_df)

    shap_vals, base_val, _, _ = compute_shap_explanation(pipeline, X_df)

    # Calculate reconstructed margin
    reconstructed_margin = base_val + shap_vals.sum(axis=1)

    # Model margin prediction (booster decision function)
    if hasattr(model, "predict"):
        try:
            actual_margin = model.get_booster().inplace_predict(X_trans, output_margin=True)
        except Exception:
            actual_probs = pipeline.predict_proba(X_df)[:, 1]
            actual_margin = np.log(actual_probs / (1.0 - actual_probs + 1e-12))
    else:
        actual_probs = pipeline.predict_proba(X_df)[:, 1]
        actual_margin = np.log(actual_probs / (1.0 - actual_probs + 1e-12))

    max_diff = float(np.max(np.abs(reconstructed_margin - actual_margin)))
    is_valid = max_diff <= atol

    # Also verify probability reconstruction: sigmoid(reconstructed) == predict_proba
    reconstructed_probs = expit(reconstructed_margin)
    actual_probs = pipeline.predict_proba(X_df)[:, 1]
    max_prob_diff = float(np.max(np.abs(reconstructed_probs - actual_probs)))

    return {
        "is_valid": is_valid,
        "max_margin_error": round(max_diff, 6),
        "max_probability_error": round(max_prob_diff, 6),
        "tolerance": atol,
        "n_samples": len(X_df),
        "base_value_logit": round(base_val, 4),
        "base_value_prob": round(float(expit(base_val)), 4),
    }


# ---------------------------------------------------------------------------
# Individual Applicant Explanation
# ---------------------------------------------------------------------------


def explain_loan_application(
    pipeline: Pipeline,
    loan_record: pd.Series | pd.DataFrame,
    feature_columns: list[str],
    top_n: int = 5,
) -> dict[str, Any]:
    """Generate an explainable risk breakdown for a single loan application.

    Parameters
    ----------
    pipeline:
        Fitted XGBoost pipeline.
    loan_record:
        Single row Series or single-row DataFrame.
    feature_columns:
        Expected input feature columns.
    top_n:
        Number of top risk factors to report.

    Returns
    -------
    dict
        Comprehensive explanation containing predicted probability, base risk,
        top positive risk drivers, top negative risk drivers, and feature contributions.
    """
    if isinstance(loan_record, pd.Series):
        record_df = pd.DataFrame([loan_record])
    else:
        record_df = loan_record.copy()

    # Reindex to match expected feature columns
    record_df = record_df.reindex(columns=feature_columns)

    # Compute prediction probability
    proba = float(pipeline.predict_proba(record_df)[0, 1])

    # Compute SHAP values for single sample
    shap_vals, base_val, feature_names, X_trans = compute_shap_explanation(pipeline, record_df)
    sample_shap = shap_vals[0]
    sample_trans = X_trans[0]

    base_prob = float(expit(base_val))

    # Build feature contribution list
    contributions = []
    for f_name, f_val, s_val in zip(feature_names, sample_trans, sample_shap):
        contributions.append({
            "feature": f_name,
            "feature_value": float(f_val),
            "shap_logit": float(s_val),
            "abs_shap": float(abs(s_val)),
            "direction": "Risk Driver (+)" if s_val > 0 else "Risk Reducer (-)",
        })

    # Sort contributions by absolute magnitude
    contributions = sorted(contributions, key=lambda x: x["abs_shap"], reverse=True)

    # Separate positive (increases default risk) and negative (decreases default risk)
    pos_drivers = [c for c in contributions if c["shap_logit"] > 0][:top_n]
    neg_drivers = [c for c in contributions if c["shap_logit"] < 0][:top_n]

    # Format human readable descriptions
    def _format_description(driver: dict[str, Any]) -> str:
        name = driver["feature"]
        val = driver["feature_value"]
        if "debtinc" in name:
            return f"Debt-to-income ratio ({val:.1f}%)"
        if "delinq" in name:
            return f"Delinquent credit lines ({int(val)} recorded)"
        if "derog" in name:
            return f"Derogatory credit reports ({int(val)} recorded)"
        if "clage" in name:
            return f"Credit history age ({val:.1f} years)"
        if "value" in name or "loan" in name or "mort" in name:
            return f"{name} ({val:.2f})"
        return f"{name} = {val:.2f}"

    return {
        "predicted_default_probability": round(proba, 4),
        "predicted_default_pct": f"{proba * 100:.1f}%",
        "base_portfolio_risk_pct": f"{base_prob * 100:.1f}%",
        "decision": "HIGH RISK / DEFAULT LIKELY" if proba >= 0.50 else "LOW RISK / APPROVED",
        "top_positive_risk_factors": [
            {
                "feature": d["feature"],
                "description": _format_description(d),
                "shap_impact": round(d["shap_logit"], 4),
            }
            for d in pos_drivers
        ],
        "top_negative_risk_factors": [
            {
                "feature": d["feature"],
                "description": _format_description(d),
                "shap_impact": round(d["shap_logit"], 4),
            }
            for d in neg_drivers
        ],
        "all_feature_contributions": [
            {
                "feature": c["feature"],
                "feature_value": round(c["feature_value"], 4),
                "shap_impact": round(c["shap_logit"], 4),
                "direction": c["direction"],
            }
            for c in contributions
        ],
    }


# ---------------------------------------------------------------------------
# Logistic Regression Interpretability
# ---------------------------------------------------------------------------


def extract_logistic_coefficients(
    pipeline: Pipeline, X_sample: pd.DataFrame
) -> pd.DataFrame:
    """Extract fitted Logistic Regression log-odds coefficients and odds ratios.

    Parameters
    ----------
    pipeline:
        Fitted Pipeline with 'preprocessor' and LogisticRegression 'model'.
    X_sample:
        Sample DataFrame matching input features.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, coefficient, odds_ratio,
        odds_direction, abs_coefficient.
    """
    model = pipeline.named_steps["model"]
    feature_names = get_preprocessed_feature_names(pipeline, X_sample)

    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)

    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coefficient": np.round(coefs, 4),
        "odds_ratio": np.round(odds_ratios, 4),
        "odds_direction": [
            "Increases Default Risk (+)" if c > 0 else "Decreases Default Risk (-)"
            for c in coefs
        ],
        "abs_coefficient": np.round(np.abs(coefs), 4),
    })

    return df_coef.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualization Generator Functions
# ---------------------------------------------------------------------------


def plot_global_shap_importance(
    shap_values: np.ndarray, feature_names: list[str], output_path: Path, top_n: int = 15
) -> None:
    """Generate and save SHAP Feature Importance bar chart (Mean Absolute SHAP Value)."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values(by="mean_abs_shap", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_imp["feature"], df_imp["mean_abs_shap"], color="#1f77b4", edgecolor="none")
    ax.set_xlabel("Mean |SHAP Value| (Average Impact on Model Output Magnitude)", fontsize=10)
    ax.set_title(f"SHAP Feature Importance — Top {top_n} Features (XGBoost)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
                va="center", ha="left", fontsize=9, color="#333333")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved SHAP Feature Importance plot → {output_path}")


def plot_xgboost_importance_comparison(
    pipeline: Pipeline, X_sample: pd.DataFrame, output_path: Path, top_n: int = 15
) -> None:
    """Generate and save XGBoost model gain importance plot."""
    model = pipeline.named_steps["model"]
    feature_names = get_preprocessed_feature_names(pipeline, X_sample)
    importances = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_imp["feature"], df_imp["importance"], color="#2ca02c", edgecolor="none")
    ax.set_xlabel("XGBoost Model Feature Gain (Split Contribution)", fontsize=10)
    ax.set_title(f"XGBoost Built-in Feature Importance — Top {top_n} Features", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
                va="center", ha="left", fontsize=9, color="#333333")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved XGBoost Feature Importance plot → {output_path}")


def plot_shap_summary(
    pipeline: Pipeline, X_df: pd.DataFrame, output_path: Path, max_display: int = 15
) -> None:
    """Generate and save SHAP summary beeswarm plot."""
    shap_vals, _, feature_names, X_trans = compute_shap_explanation(pipeline, X_df)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals,
        X_trans,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Summary Plot — Feature Value vs. Default Risk Impact", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP Summary plot → {output_path}")
