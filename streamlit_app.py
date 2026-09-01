"""
CreditRisk AI — Credit Default Prediction & Portfolio Risk Analytics.

Professional 5-Tab Streamlit Dashboard:
  Tab 1: Portfolio Overview
  Tab 2: Model Performance
  Tab 3: Explainability
  Tab 4: Expected Credit Loss (ECL)
  Tab 5: Loan Risk Scoring
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gg
import streamlit as st

# Project relative paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hmeq.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"
EXPLAIN_DIR = ARTIFACTS_DIR / "explainability"
PORTFOLIO_DIR = ARTIFACTS_DIR / "portfolio"

# Backend modules
from src.dataset_loader import load_hmeq
from src.ecl import (
    DEFAULT_LGD,
    calculate_ecl_by_category,
    calculate_ecl_by_risk_segment,
    calculate_portfolio_ecl,
    compute_portfolio_kpis,
)
from src.explainability import (
    explain_loan_application,
    extract_logistic_coefficients,
)
from src.features import engineer_features
from src.modeling import load_artifacts
from src.scoring import score_portfolio


# ---------------------------------------------------------------------------
# Streamlit Configuration & Custom Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CreditRisk AI — Risk Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply sleek CSS styling
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stMetric { background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 12px; border-radius: 8px; }
    .css-1544g2n { padding-top: 1rem; }
    .badge-approved { background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-risk { background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model_pipeline() -> tuple[Any, dict[str, Any]]:
    """Load trained model pipeline and metadata from artifacts/."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error(
            f"Model artifacts missing at `{MODEL_PATH}`. Please run:\n"
            "```bash\npython scripts/train_model.py\n```"
        )
        st.stop()
    return load_artifacts(MODEL_PATH, METADATA_PATH)


@st.cache_data(show_spinner=False)
def load_raw_dataset(dataset_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load raw HMEQ dataset using dataset_loader."""
    if not dataset_path.exists():
        st.error(
            f"Dataset missing at `{dataset_path}`. Please run:\n"
            "```powershell\nInvoke-WebRequest -Uri \"https://raw.githubusercontent.com/sassoftware/sas-viya-dmml-pipelines/master/data/hmeq.csv\" -OutFile \"data\\raw\\hmeq.csv\"\n```"
        )
        st.stop()
    return load_hmeq(dataset_path)


@st.cache_data(show_spinner=False)
def load_comparison_data() -> dict[str, Any] | None:
    """Load model_comparison.json artifact if present."""
    if COMPARISON_PATH.exists():
        try:
            return json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("💳 CreditRisk AI")
st.sidebar.caption("Portfolio Risk Analytics Platform")
st.sidebar.divider()

# Sidebar: Controls
st.sidebar.subheader("⚙️ Portfolio Controls")

data_source = st.sidebar.radio(
    "Dataset Source",
    ["Standard HMEQ (5,960 loans)", "Upload Portfolio CSV"],
    index=0,
)

uploaded_file = None
if data_source == "Upload Portfolio CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.divider()
st.sidebar.subheader("🎛️ Risk Model Controls")

decision_threshold = st.sidebar.slider(
    "Decision Threshold (PD Cutoff)",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Probability cutoff for classifying default risk (BAD=1 vs BAD=0).",
)

lgd_assumption = st.sidebar.slider(
    "Loss Given Default (LGD)",
    min_value=0.10,
    max_value=0.90,
    value=DEFAULT_LGD,
    step=0.05,
    help="Assumed percentage loss on defaulted exposure ($ECL = PD \\times LGD \\times EAD$).",
)

st.sidebar.divider()

# Model Artifact Readiness Status
pipeline, metadata = load_model_pipeline()
feature_columns = metadata.get("feature_columns", [])
comp_data = load_comparison_data()

st.sidebar.subheader("📦 System Status")
st.sidebar.markdown(f"**Active Model:** `{metadata.get('model_name', 'XGBoost').upper()}`")
st.sidebar.markdown(f"**Trained Features:** `{len(feature_columns)}`")
st.sidebar.markdown(f"**Training Rows:** `{metadata.get('rows_used', 5960):,}`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Artifact Readiness:**")
st.sidebar.markdown("✅ `model.joblib` loaded")
st.sidebar.markdown("✅ `model_comparison.json` " if comp_data else "⚠️ `model_comparison.json` missing")
st.sidebar.markdown("✅ `explainability/` artifacts " if (EXPLAIN_DIR / "shap_feature_importance.png").exists() else "⚠️ Explainability images missing")


# ---------------------------------------------------------------------------
# Portfolio Data Ingestion & Scoring
# ---------------------------------------------------------------------------

if data_source == "Upload Portfolio CSV" and uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = load_raw_dataset()

# Score portfolio dynamically based on selected threshold and LGD
scored_portfolio = score_portfolio(
    raw_df=raw_df,
    pipeline=pipeline,
    feature_columns=feature_columns,
    lgd_val=lgd_assumption,
    threshold=decision_threshold,
)

# Compute portfolio KPIs
portfolio_kpis = compute_portfolio_kpis(scored_portfolio)
segment_summary = calculate_ecl_by_risk_segment(scored_portfolio)


# ---------------------------------------------------------------------------
# Dashboard Header
# ---------------------------------------------------------------------------

st.title("CreditRisk AI — Credit Default Prediction & Portfolio Risk Analytics")
st.caption(
    "An end-to-end machine learning platform for credit default prediction, explainability, "
    "and expected credit loss (ECL) portfolio analytics."
)

# Five-Tab Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Portfolio Overview",
    "📈 Model Performance",
    "🔍 Explainability",
    "🛡️ Expected Credit Loss",
    "🎯 Loan Risk Scoring",
])


# ===========================================================================
# TAB 1 — PORTFOLIO OVERVIEW
# ===========================================================================
with tab1:
    st.subheader("Executive Portfolio Summary")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Loans", f"{portfolio_kpis['portfolio_loan_count']:,}")
    k2.metric("Total Exposure (EAD)", f"${portfolio_kpis['total_exposure_ead']:,.0f}")
    k3.metric("Weighted Avg PD", f"{portfolio_kpis['weighted_average_pd'] * 100:.2f}%")
    k4.metric("Total ECL", f"${portfolio_kpis['total_expected_credit_loss_ecl']:,.0f}")
    k5.metric("Expected Loss Rate", portfolio_kpis["portfolio_loss_rate_pct"])
    k6.metric("High Risk EAD %", f"{portfolio_kpis['high_risk_exposure_pct']:.1f}%")

    if portfolio_kpis["observed_historical_bad_rate"] != "N/A":
        st.info(
            f"ℹ️ **Historical Dataset Default Rate (BAD=1):** `{float(portfolio_kpis['observed_historical_bad_rate']) * 100:.2f}%` "
            f"| **Active LGD Assumption:** `{lgd_assumption:.0%}` "
            f"| **Decision Threshold:** `{decision_threshold:.2f}`"
        )

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        # Chart 1: Risk Segment Distribution (Loans)
        fig_seg_count = px.bar(
            segment_summary,
            x="risk_segment",
            y="loan_count",
            color="risk_segment",
            color_discrete_map={"Low": "#2ca02c", "Moderate": "#ff7f0e", "High": "#d62728", "Very High": "#9467bd"},
            title="Loan Count by Risk Segment",
            text="loan_count",
        )
        fig_seg_count.update_layout(showlegend=False, xaxis_title="Risk Segment", yaxis_title="Number of Loans")
        st.plotly_chart(fig_seg_count, use_container_width=True)

        # Chart 3: Expected Credit Loss by Risk Segment
        fig_seg_ecl = px.bar(
            segment_summary,
            x="risk_segment",
            y="total_ecl",
            color="risk_segment",
            color_discrete_map={"Low": "#2ca02c", "Moderate": "#ff7f0e", "High": "#d62728", "Very High": "#9467bd"},
            title="Expected Credit Loss (ECL $) by Risk Segment",
            text_auto=".2s",
        )
        fig_seg_ecl.update_layout(showlegend=False, xaxis_title="Risk Segment", yaxis_title="Total ECL ($)")
        st.plotly_chart(fig_seg_ecl, use_container_width=True)

    with c2:
        # Chart 2: Portfolio Exposure (EAD) by Risk Segment
        fig_seg_ead = px.bar(
            segment_summary,
            x="risk_segment",
            y="total_ead",
            color="risk_segment",
            color_discrete_map={"Low": "#2ca02c", "Moderate": "#ff7f0e", "High": "#d62728", "Very High": "#9467bd"},
            title="Portfolio Exposure (EAD $) by Risk Segment",
            text_auto=".2s",
        )
        fig_seg_ead.update_layout(showlegend=False, xaxis_title="Risk Segment", yaxis_title="Total EAD ($)")
        st.plotly_chart(fig_seg_ead, use_container_width=True)

        # Chart 4: Default Probability Distribution
        fig_pd_dist = px.histogram(
            scored_portfolio,
            x="default_probability",
            nbins=40,
            title="Portfolio Default Probability Distribution",
            color_discrete_sequence=["#1f77b4"],
        )
        fig_pd_dist.add_vline(
            x=decision_threshold, line_dash="dash", line_color="red",
            annotation_text=f"Cutoff ({decision_threshold:.2f})", annotation_position="top right"
        )
        fig_pd_dist.update_layout(xaxis_title="Predicted Default Probability (PD)", yaxis_title="Loan Count")
        st.plotly_chart(fig_pd_dist, use_container_width=True)

    st.subheader("⚠️ Top 20 Riskiest Loans in Portfolio")
    top_20 = scored_portfolio.sort_values(by="default_probability", ascending=False).head(20)
    disp_cols = [c for c in ["loan", "value", "mortdue", "debtinc", "default_probability", "risk_segment", "lgd", "ead", "ecl"] if c in top_20.columns]
    st.dataframe(top_20[disp_cols].style.format({
        "loan": "${:,.0f}", "value": "${:,.0f}", "mortdue": "${:,.0f}",
        "debtinc": "{:.1f}%", "default_probability": "{:.2%}",
        "lgd": "{:.0%}", "ead": "${:,.0f}", "ecl": "${:,.2f}"
    }), use_container_width=True)


# ===========================================================================
# TAB 2 — MODEL PERFORMANCE
# ===========================================================================
with tab2:
    st.subheader("Model Performance & Comparison")

    if comp_data is None:
        st.warning(
            "Model comparison artifact (`artifacts/model_comparison.json`) is not found. "
            "Run `python scripts/compare_models.py` to generate side-by-side performance benchmarks."
        )
    else:
        # Build Comparison Table
        model_names = list(comp_data["models"].keys())
        metrics_rows = []
        for name in model_names:
            m = comp_data["models"][name]["metrics_at_0.50_threshold"]
            ks = comp_data["models"][name]["ks_statistic"]
            ap = comp_data["models"][name]["pr_curve"]["average_precision"]
            cv = comp_data["models"][name]["cross_validation"]

            metrics_rows.append({
                "Model": name,
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "Accuracy": f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1 Score": f"{m['f1']:.4f}",
                "KS Statistic": f"{ks:.4f}",
                "Average Precision": f"{ap:.4f}",
                "5-Fold CV Mean AUC": f"{cv['mean_roc_auc']:.4f} ± {cv['std_roc_auc']:.4f}",
            })

        st.markdown("##### 🏆 Model Benchmark Comparison (Shared Test Set, N = 1,192)")
        st.table(pd.DataFrame(metrics_rows).set_index("Model"))

        col_roc, col_pr = st.columns(2)

        with col_roc:
            # Interactive ROC Curve
            fig_roc = gg.Figure()
            fig_roc.add_trace(gg.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="black"), name="Random (0.50)"))
            colors = ["#1f77b4", "#2ca02c"]
            for idx, name in enumerate(model_names):
                roc = comp_data["models"][name]["roc_curve"]
                auc = comp_data["models"][name]["metrics_at_0.50_threshold"]["roc_auc"]
                fig_roc.add_trace(gg.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"{name} (AUC={auc:.4f})", line=dict(color=colors[idx % 2], width=2)))
            fig_roc.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_pr:
            # Interactive PR Curve
            fig_pr = gg.Figure()
            base_rate = comp_data["class_balance_test"]["bad_1_pct"] / 100
            fig_pr.add_trace(gg.Scatter(x=[0, 1], y=[base_rate, base_rate], mode="lines", line=dict(dash="dash", color="black"), name=f"Baseline ({base_rate:.1%})"))
            for idx, name in enumerate(model_names):
                pr = comp_data["models"][name]["pr_curve"]
                ap = pr["average_precision"]
                fig_pr.add_trace(gg.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name=f"{name} (AP={ap:.4f})", line=dict(color=colors[idx % 2], width=2)))
            fig_pr.update_layout(title="Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision")
            st.plotly_chart(fig_pr, use_container_width=True)

        # Interactive Threshold Sensitivity Analysis
        st.subheader("🎛️ Interactive Threshold Sensitivity Analysis")
        st.caption("Evaluate business metrics across decision thresholds for the active model.")

        active_model_name = "XGBoost" if "XGBoost" in model_names else model_names[0]
        thresh_data = comp_data["models"][active_model_name]["threshold_analysis"]
        df_thresh = pd.DataFrame(thresh_data)

        st.dataframe(df_thresh.style.format({
            "threshold": "{:.2f}", "precision": "{:.4f}", "recall": "{:.4f}",
            "f1": "{:.4f}", "predicted_default_rate": "{:.1%}"
        }), use_container_width=True)

        st.info(
            "💡 **Business Risk Trade-Off:**\n"
            "- **Lower Threshold (< 0.30):** High Recall (detects ~80%+ defaults), but increases false rejections.\n"
            "- **Higher Threshold (> 0.50):** High Precision (fewer false rejections), but misses legitimate defaults."
        )


# ===========================================================================
# TAB 3 — EXPLAINABILITY
# ===========================================================================
with tab3:
    st.subheader("Model Explainability & SHAP Feature Attribution")

    exp_tab1, exp_tab2 = st.tabs(["🌐 Global Feature Importance", "👤 Individual Applicant Breakdown"])

    with exp_tab1:
        img_shap = EXPLAIN_DIR / "shap_feature_importance.png"
        img_xgb = EXPLAIN_DIR / "xgboost_feature_importance.png"
        img_summary = EXPLAIN_DIR / "shap_summary.png"

        g1, g2 = st.columns(2)
        with g1:
            if img_shap.exists():
                st.image(str(img_shap), caption="SHAP Global Feature Importance (Mean Absolute SHAP Value)")
            else:
                st.info("Run `python scripts/verify_phase3.py` to generate global SHAP charts.")

        with g2:
            if img_xgb.exists():
                st.image(str(img_xgb), caption="XGBoost Built-in Feature Importance (Gain)")
            else:
                st.info("XGBoost importance image missing.")

        if img_summary.exists():
            st.markdown("---")
            st.image(str(img_summary), caption="SHAP Summary Beeswarm Plot — Feature Value vs Risk Impact")

    with exp_tab2:
        st.subheader("Single Loan SHAP Risk Breakdown")
        loan_indices = list(scored_portfolio.index)
        selected_idx = st.selectbox("Select Applicant Record Index", loan_indices, index=0)

        selected_record = scored_portfolio.loc[selected_idx]

        explanation = explain_loan_application(
            pipeline=pipeline,
            loan_record=selected_record,
            feature_columns=feature_columns,
            top_n=5,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Applicant Loan Amount", f"${selected_record.get('loan', 0):,.0f}")
        m2.metric("Predicted Default Probability", explanation["predicted_default_pct"])
        m3.metric("Risk Segment", selected_record.get("risk_segment", "N/A"))
        m4.metric("Expected Credit Loss (ECL)", f"${selected_record.get('ecl', 0):,.2f}")

        if explanation["predicted_default_probability"] >= decision_threshold:
            st.error(f"🚨 **Underwriting Decision:** {explanation['decision']}")
        else:
            st.success(f"✅ **Underwriting Decision:** {explanation['decision']}")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("##### 📈 Top Risk Drivers (Increasing Default Probability)")
            for item in explanation["top_positive_risk_factors"]:
                st.write(f"- **{item['description']}** (SHAP impact: `+{item['shap_impact']:.4f}` logit)")

        with r2:
            st.markdown("##### 📉 Top Risk Mitigators (Decreasing Default Probability)")
            for item in explanation["top_negative_risk_factors"]:
                st.write(f"- **{item['description']}** (SHAP impact: `{item['shap_impact']:.4f}` logit)")


# ===========================================================================
# TAB 4 — EXPECTED CREDIT LOSS (ECL)
# ===========================================================================
with tab4:
    st.subheader("Expected Credit Loss (ECL) Analytics Engine")

    st.markdown(
        "$$\\text{Expected Credit Loss (ECL)} = \\text{PD} \\times \\text{LGD} \\times \\text{EAD}$$"
    )

    st.caption(
        "• **PD (Probability of Default):** Model-predicted default probability.\n"
        "• **LGD (Loss Given Default):** User-adjustable assumption (current sidebar setting).\n"
        "• **EAD (Exposure at Default):** Loan request amount ($LOAN) proxy."
    )

    st.divider()

    # Recalculate KPIs with dynamic LGD
    dynamic_ecl_df = calculate_portfolio_ecl(scored_portfolio, pd_col="default_probability", lgd_val=lgd_assumption, ead_col="loan")
    dyn_kpis = compute_portfolio_kpis(dynamic_ecl_df)
    dyn_segments = calculate_ecl_by_risk_segment(dynamic_ecl_df)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total Exposure (EAD)", f"${dyn_kpis['total_exposure_ead']:,.0f}")
    e2.metric("Total ECL", f"${dyn_kpis['total_expected_credit_loss_ecl']:,.0f}")
    e3.metric("Expected Loss Rate", dyn_kpis["portfolio_loss_rate_pct"])
    e4.metric("Weighted Average PD", f"{dyn_kpis['weighted_average_pd'] * 100:.2f}%")

    st.subheader("ECL & Exposure Breakdown by Risk Segment")
    st.dataframe(dyn_segments.style.format({
        "pct_of_loans": "{:.1f}%", "total_ead": "${:,.0f}", "pct_of_ead": "{:.1f}%",
        "avg_pd": "{:.2%}", "total_ecl": "${:,.2f}", "avg_ecl": "${:,.2f}", "pct_of_ecl": "{:.1f}%"
    }), use_container_width=True)

    st.subheader("Portfolio Concentration Analysis")
    conc_tab1, conc_tab2 = st.tabs(["👔 By Occupation (JOB)", "🎯 By Purpose (REASON)"])

    with conc_tab1:
        job_conc = calculate_ecl_by_category(dynamic_ecl_df, category_col="job")
        st.dataframe(job_conc.style.format({
            "total_ead": "${:,.0f}", "avg_pd": "{:.2%}", "total_ecl": "${:,.2f}", "pct_of_total_ecl": "{:.1f}%"
        }), use_container_width=True)

    with conc_tab2:
        reason_conc = calculate_ecl_by_category(dynamic_ecl_df, category_col="reason")
        st.dataframe(reason_conc.style.format({
            "total_ead": "${:,.0f}", "avg_pd": "{:.2%}", "total_ecl": "${:,.2f}", "pct_of_total_ecl": "{:.1f}%"
        }), use_container_width=True)

    st.warning(
        "⚠️ **Disclaimer:** This Expected Credit Loss calculation is a simplified analytical implementation "
        "and is not intended for regulatory or production banking use."
    )


# ===========================================================================
# TAB 5 — LOAN RISK SCORING (SINGLE APPLICANT INTERFACE)
# ===========================================================================
with tab5:
    st.subheader("Single Loan Risk Scoring & Underwriting Analysis")

    with st.form("single_loan_form"):
        st.markdown("##### Enter Loan Application Attributes")
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            inp_loan = st.number_input("LOAN ($)", min_value=1000, max_value=100000, value=15000, step=1000)
            inp_mortdue = st.number_input("MORTDUE ($)", min_value=0, max_value=500000, value=50000, step=5000)
            inp_value = st.number_input("VALUE ($)", min_value=1000, max_value=1000000, value=85000, step=5000)

        with f2:
            inp_reason = st.selectbox("REASON", ["DebtCon", "HomeImp"], index=0)
            inp_job = st.selectbox("JOB", ["Other", "ProfExe", "Office", "Mgr", "Self", "Sales"], index=0)
            inp_yoj = st.number_input("YOJ (Years at Job)", min_value=0.0, max_value=50.0, value=6.0, step=0.5)

        with f3:
            inp_derog = st.number_input("DEROG (Derogatory Reports)", min_value=0, max_value=20, value=0, step=1)
            inp_delinq = st.number_input("DELINQ (Delinquent Lines)", min_value=0, max_value=20, value=0, step=1)
            inp_clage = st.number_input("CLAGE (Credit Age in Months)", min_value=0.0, max_value=1000.0, value=180.0, step=10.0)

        with f4:
            inp_ninq = st.number_input("NINQ (Credit Inquiries)", min_value=0, max_value=20, value=1, step=1)
            inp_clno = st.number_input("CLNO (Total Credit Lines)", min_value=1, max_value=100, value=20, step=1)
            inp_debtinc = st.number_input("DEBTINC (Debt-to-Income Ratio %)", min_value=0.0, max_value=100.0, value=32.0, step=1.0)

        submit_button = st.form_submit_button("🎯 Analyze Loan Risk", use_container_width=True)

    if submit_button:
        # Create single row DataFrame
        single_raw = pd.DataFrame([{
            "loan": inp_loan,
            "mortdue": inp_mortdue,
            "value": inp_value,
            "reason": inp_reason,
            "job": inp_job,
            "yoj": inp_yoj,
            "derog": inp_derog,
            "delinq": inp_delinq,
            "clage": inp_clage,
            "ninq": inp_ninq,
            "clno": inp_clno,
            "debtinc": inp_debtinc,
        }])

        single_scored = score_portfolio(
            raw_df=single_raw,
            pipeline=pipeline,
            feature_columns=feature_columns,
            lgd_val=lgd_assumption,
            threshold=decision_threshold,
        )

        single_pd = float(single_scored["default_probability"].iloc[0])
        single_segment = single_scored["risk_segment"].iloc[0]
        single_ecl = float(single_scored["ecl"].iloc[0])

        st.divider()
        st.subheader("Analysis Results")

        res1, res2, res3, res4 = st.columns(4)
        res1.metric("Default Probability", f"{single_pd:.2%}")
        res2.metric("Risk Segment", single_segment)
        res3.metric("Expected Credit Loss (ECL)", f"${single_ecl:,.2f}")
        res4.metric("LGD / EAD", f"{lgd_assumption:.0%} / ${inp_loan:,.0f}")

        if single_pd >= decision_threshold:
            st.error(f"🚨 **UNDERWRITING DECISION: REJECTED / HIGH RISK** (PD {single_pd:.1%} >= Cutoff {decision_threshold:.1%})")
        else:
            st.success(f"✅ **UNDERWRITING DECISION: APPROVED / LOW RISK** (PD {single_pd:.1%} < Cutoff {decision_threshold:.1%})")

        # Generate SHAP explanation for single input
        single_exp = explain_loan_application(
            pipeline=pipeline,
            loan_record=single_scored.iloc[0],
            feature_columns=feature_columns,
            top_n=5,
        )

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("##### 📈 Top Positive Risk Drivers (+)")
            for d in single_exp["top_positive_risk_factors"]:
                st.write(f"- **{d['description']}** (`+{d['shap_impact']:.4f}` logit)")

        with s_col2:
            st.markdown("##### 📉 Top Negative Risk Mitigators (-)")
            for d in single_exp["top_negative_risk_factors"]:
                st.write(f"- **{d['description']}** (`{d['shap_impact']:.4f}` logit)")
