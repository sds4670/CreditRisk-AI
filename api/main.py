"""
Production FastAPI Credit Risk Scoring & Portfolio Analytics API — Phase 6.

Endpoints
---------
GET  /health            - API health & artifact readiness (Public)
GET  /model-info        - Active model metrics, KS, features, & metadata
POST /score             - Score single or batch loan applications
POST /portfolio-summary - Compute portfolio-level EAD, PD, ECL & risk distribution

Authentication
--------------
Optional API Key authentication via ``X-API-Key`` header.
Configured via ``CREDITRISK_API_KEY`` environment variable.
If ``CREDITRISK_API_KEY`` is not set, auth is disabled for local dev.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# Project relative paths
ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"
DATA_PATH = ROOT_DIR / "data" / "raw" / "hmeq.csv"

# Backend modules
from src.ecl import DEFAULT_LGD, calculate_ecl_by_risk_segment, compute_portfolio_kpis
from src.modeling import load_artifacts
from src.scoring import score_portfolio


# ---------------------------------------------------------------------------
# FastAPI App & Security Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CreditRisk AI — Credit Risk Scoring API",
    description=(
        "Production REST API for credit default prediction, expected credit loss (ECL) "
        "analytics, risk segmentation, and model metadata."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """Validate X-API-Key header against CREDITRISK_API_KEY environment variable.

    If CREDITRISK_API_KEY is not set in environment, auth is disabled for local dev.
    """
    expected_key = os.getenv("CREDITRISK_API_KEY")
    if not expected_key:
        return api_key  # Auth disabled in local development mode

    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Pass 'X-API-Key' header.",
        )
    return api_key


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class LoanApplicationRecord(BaseModel):
    """Pydantic model representing a single credit application (HMEQ attributes)."""

    loan: float = Field(..., example=15000.0, description="Requested loan amount ($)")
    mortdue: float = Field(..., example=50000.0, description="Amount due on existing mortgage ($)")
    value: float = Field(..., example=85000.0, description="Current property value ($)")
    reason: str = Field("DebtCon", example="DebtCon", description="Loan purpose: DebtCon | HomeImp")
    job: str = Field("Office", example="Office", description="Occupational category: Other, ProfExe, Office, Mgr, Self, Sales")
    yoj: float = Field(6.0, example=6.0, description="Years at present job")
    derog: float = Field(0.0, example=0.0, description="Number of major derogatory reports")
    delinq: float = Field(0.0, example=0.0, description="Number of delinquent credit lines")
    clage: float = Field(180.0, example=180.0, description="Age of oldest credit line (months)")
    ninq: float = Field(1.0, example=1.0, description="Number of recent credit inquiries")
    clno: float = Field(20.0, example=20.0, description="Total number of credit lines")
    debtinc: float = Field(32.0, example=32.0, description="Debt-to-income ratio (%)")
    lgd: Optional[float] = Field(DEFAULT_LGD, example=0.45, description="Optional custom Loss Given Default assumption (0.10 to 0.90)")

    class Config:
        json_schema_extra = {
            "example": {
                "loan": 15000.0,
                "mortdue": 50000.0,
                "value": 85000.0,
                "reason": "DebtCon",
                "job": "Office",
                "yoj": 6.0,
                "derog": 0.0,
                "delinq": 0.0,
                "clage": 180.0,
                "ninq": 1.0,
                "clno": 20.0,
                "debtinc": 32.0,
                "lgd": 0.45,
            }
        }


class SingleScoreRequest(BaseModel):
    record: LoanApplicationRecord
    threshold: Optional[float] = Field(0.50, example=0.50, description="Decision threshold cutoff for default classification")


class BatchScoreRequest(BaseModel):
    records: List[LoanApplicationRecord]
    threshold: Optional[float] = Field(0.50, example=0.50, description="Decision threshold cutoff")
    lgd: Optional[float] = Field(DEFAULT_LGD, example=0.45, description="Loss Given Default assumption across batch")


# ---------------------------------------------------------------------------
# Artifact Loader Helper
# ---------------------------------------------------------------------------


def get_pipeline_and_metadata() -> tuple[Any, dict[str, Any]]:
    """Load model pipeline and metadata from disk without auto-retraining."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts missing. Run 'python scripts/train_model.py' to generate model.joblib.",
        )
    return load_artifacts(MODEL_PATH, METADATA_PATH)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
def health_check() -> dict[str, Any]:
    """Public health check endpoint returning API status, model info, and artifact availability."""
    model_exists = MODEL_PATH.exists()
    comp_exists = COMPARISON_PATH.exists()
    data_exists = DATA_PATH.exists()

    model_name = "unknown"
    feature_count = 0
    if model_exists and METADATA_PATH.exists():
        try:
            _, meta = get_pipeline_and_metadata()
            model_name = meta.get("model_name", "xgboost")
            feature_count = len(meta.get("feature_columns", []))
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "CreditRisk AI Scoring API",
        "version": "1.0.0",
        "active_model": model_name,
        "feature_count": feature_count,
        "artifacts_available": {
            "model_joblib": model_exists,
            "model_comparison_json": comp_exists,
            "hmeq_dataset_csv": data_exists,
        },
        "auth_enabled": bool(os.getenv("CREDITRISK_API_KEY")),
    }


@app.get("/model-info", status_code=status.HTTP_200_OK, tags=["Model Info"])
def get_model_info(api_key: Optional[str] = Depends(verify_api_key)) -> dict[str, Any]:
    """Return model performance metrics, KS statistic, ROC-AUC, and feature list."""
    pipeline, meta = get_pipeline_and_metadata()

    roc_auc = meta.get("metrics", {}).get("roc_auc")
    comp_data = None
    if COMPARISON_PATH.exists():
        try:
            import json
            comp_data = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    ks_stat = None
    cv_info = None
    if comp_data and "models" in comp_data:
        active_name = "XGBoost" if "XGBoost" in comp_data["models"] else list(comp_data["models"].keys())[0]
        ks_stat = comp_data["models"][active_name].get("ks_statistic")
        cv_info = comp_data["models"][active_name].get("cross_validation")

    return {
        "active_model": meta.get("model_name", "xgboost"),
        "metrics": meta.get("metrics", {}),
        "holdout_roc_auc": roc_auc,
        "ks_statistic": ks_stat,
        "cross_validation_5fold": cv_info,
        "training_rows": meta.get("rows_used"),
        "feature_count": len(meta.get("feature_columns", [])),
        "feature_columns": meta.get("feature_columns", []),
        "dataset_path": meta.get("dataset_path"),
    }


@app.post("/score", status_code=status.HTTP_200_OK, tags=["Scoring"])
def score_loans(
    request: BatchScoreRequest, api_key: Optional[str] = Depends(verify_api_key)
) -> dict[str, Any]:
    """Score single or batch loan applications.

    Returns default probability, risk segment, EAD, LGD, ECL, and underwriting decision.
    """
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request 'records' list cannot be empty.",
        )

    pipeline, meta = get_pipeline_and_metadata()
    feature_cols = meta["feature_columns"]

    # Convert Pydantic records to DataFrame
    raw_dicts = [r.dict() for r in request.records]
    raw_df = pd.DataFrame(raw_dicts)

    # Score portfolio using src.scoring
    scored = score_portfolio(
        raw_df=raw_df,
        pipeline=pipeline,
        feature_columns=feature_cols,
        lgd_val=request.lgd or DEFAULT_LGD,
        threshold=request.threshold or 0.50,
    )

    results = []
    for idx, row in scored.iterrows():
        pd_val = float(row["default_probability"])
        is_default = bool(pd_val >= (request.threshold or 0.50))
        results.append({
            "record_index": int(idx),
            "loan_amount": float(row.get("loan", row.get("loan_amnt", 0.0))),
            "default_probability": pd_val,
            "predicted_default": is_default,
            "underwriting_decision": "HIGH RISK / REJECT" if is_default else "LOW RISK / APPROVE",
            "risk_segment": str(row["risk_segment"]),
            "ead": float(row["ead"]),
            "lgd": float(row["lgd"]),
            "ecl": float(row["ecl"]),
        })

    return {
        "records_scored": len(results),
        "decision_threshold": request.threshold or 0.50,
        "assumed_lgd": request.lgd or DEFAULT_LGD,
        "average_default_probability": round(float(scored["default_probability"].mean()), 4),
        "total_ecl": round(float(scored["ecl"].sum()), 2),
        "predictions": results,
    }


@app.post("/portfolio-summary", status_code=status.HTTP_200_OK, tags=["Portfolio Risk Analytics"])
def get_portfolio_summary_api(
    request: BatchScoreRequest, api_key: Optional[str] = Depends(verify_api_key)
) -> dict[str, Any]:
    """Score uploaded portfolio and return executive KPIs and risk segment distribution."""
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request 'records' list cannot be empty.",
        )

    pipeline, meta = get_pipeline_and_metadata()
    feature_cols = meta["feature_columns"]

    raw_df = pd.DataFrame([r.dict() for r in request.records])
    scored = score_portfolio(
        raw_df=raw_df,
        pipeline=pipeline,
        feature_columns=feature_cols,
        lgd_val=request.lgd or DEFAULT_LGD,
        threshold=request.threshold or 0.50,
    )

    kpis = compute_portfolio_kpis(scored)
    segments = calculate_ecl_by_risk_segment(scored)

    return {
        "portfolio_kpis": kpis,
        "risk_segment_distribution": segments.to_dict(orient="records"),
    }
