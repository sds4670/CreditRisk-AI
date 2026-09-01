"""
Model evaluation utilities for CreditRisk AI — Phase 2.

All functions in this module are **pure evaluation**: they receive fitted
pipelines and pre-split data; they never fit any statistics themselves.
This ensures evaluation results are computed strictly on held-out test data.

Public API
----------
- compute_ks_statistic      KS statistic for binary credit-risk classification
- compute_threshold_analysis  Metrics at multiple decision thresholds
- compute_calibration_table   Probability-bin calibration table (PD calibration)
- evaluate_model              Full metric suite for one fitted pipeline
- run_cv                      Stratified 5-fold CV on training data only
- run_full_comparison         Orchestrate comparison of multiple models
- plot_roc_pr_curves          Generate ROC + PR curve PNG

KS Statistic note
-----------------
The KS statistic measures the maximum separation between the cumulative
distribution of predicted scores for good loans (BAD=0) and bad/defaulted
loans (BAD=1).  A higher KS indicates better rank-ordering ability.
Industry guidance: KS > 0.40 is considered acceptable for credit scoring.

Calibration note
----------------
Calibration measures how closely predicted probabilities correspond to
observed default rates.  A perfectly calibrated model in the 0.30 bin
would see ~30% actual defaults.  This matters because predicted PDs are
used as inputs to ECL = PD × LGD × EAD.  Raw model probabilities are
NOT regulatory-grade calibrated PDs without additional post-processing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# Default decision thresholds for threshold analysis
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60)


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------


def compute_ks_statistic(y_true: np.ndarray | pd.Series, y_proba: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov (KS) statistic for binary classification.

    The KS statistic is the maximum absolute difference between the cumulative
    distribution of predicted scores for the positive class (BAD=1, defaulted)
    and the negative class (BAD=0, non-bad).  It is a standard credit-scoring
    rank-ordering metric.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0 or 1).
    y_proba:
        Predicted probability of the positive class (BAD=1).

    Returns
    -------
    float
        KS statistic in [0, 1].  Higher is better.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    # Sort by predicted probability descending (highest-risk first)
    order = np.argsort(-y_proba_arr)
    y_sorted = y_true_arr[order]

    n_pos = y_sorted.sum()       # total BAD=1
    n_neg = len(y_sorted) - n_pos  # total BAD=0

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Cumulative distribution of BAD=1 and BAD=0
    cum_bad = np.cumsum(y_sorted) / n_pos
    cum_good = np.cumsum(1 - y_sorted) / n_neg

    ks = float(np.max(np.abs(cum_bad - cum_good)))
    return round(ks, 4)


def compute_threshold_analysis(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Evaluate precision, recall, F1, and predicted default rate at each threshold.

    Business interpretation
    -----------------------
    Lower threshold → flags more borrowers as risky → higher recall but more
    false positives (legitimate borrowers incorrectly rejected).
    Higher threshold → fewer false positives but more missed defaults.
    The optimal threshold depends on the relative cost of missed defaults vs
    false rejections — a business / regulatory decision, not a purely
    statistical one.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_proba:
        Predicted probability of BAD=1.
    thresholds:
        Decision thresholds to evaluate.

    Returns
    -------
    list[dict]
        One row per threshold with keys:
        threshold, precision, recall, f1, predicted_default_rate,
        true_positives, false_positives, false_negatives, true_negatives.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": round(t, 2),
                "precision": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
                "predicted_default_rate": round(float(y_pred.mean()), 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }
        )
    return rows


def compute_calibration_table(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Build a probability-bin calibration table.

    Each bin groups observations by predicted PD.  The observed default rate
    within each bin is compared to the mean predicted probability.  A
    well-calibrated model produces rows where these two values are close.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_proba:
        Predicted probability of BAD=1.
    n_bins:
        Number of equal-width probability bins (default 10).

    Returns
    -------
    list[dict]
        One row per non-empty bin with keys:
        bin, bin_range, count, mean_predicted_pd, observed_default_rate,
        calibration_error (observed - predicted).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include right endpoint in the last bin
        if i < n_bins - 1:
            mask = (y_proba >= lo) & (y_proba < hi)
        else:
            mask = (y_proba >= lo) & (y_proba <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_proba[mask].mean())
        obs_rate = float(y_true_arr[mask].mean())
        rows.append(
            {
                "bin": i + 1,
                "bin_range": f"[{lo:.2f}, {hi:.2f})",
                "count": count,
                "mean_predicted_pd": round(mean_pred, 4),
                "observed_default_rate": round(obs_rate, 4),
                "calibration_error": round(obs_rate - mean_pred, 4),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Compute the full evaluation suite for one fitted pipeline.

    Parameters
    ----------
    name:
        Human-readable model name (e.g. ``"Logistic Regression"``).
    pipeline:
        Fitted sklearn Pipeline with a ``predict_proba`` method.
    X_test, y_test:
        Held-out test features and labels (from the shared split).
    thresholds:
        Decision thresholds for threshold analysis.

    Returns
    -------
    dict
        Structured evaluation results including metrics, curves, KS,
        calibration table, and threshold analysis.
    """
    y_true = np.asarray(y_test, dtype=int)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_50 = (y_proba >= 0.50).astype(int)

    # ── Core metrics at 0.50 threshold ────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_50, labels=[0, 1]).ravel()
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred_50)), 4),
        "precision": round(float(precision_score(y_true, y_pred_50, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred_50, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred_50, zero_division=0)), 4),
    }

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "matrix_2x2": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    # ── KS statistic ──────────────────────────────────────────────────
    ks = compute_ks_statistic(y_true, y_proba)

    # ── ROC curve data ────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_data = {
        "fpr": [round(float(x), 5) for x in fpr],
        "tpr": [round(float(x), 5) for x in tpr],
    }

    # ── Precision-Recall curve data ───────────────────────────────────
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = round(float(average_precision_score(y_true, y_proba)), 4)
    pr_data = {
        "precision": [round(float(x), 5) for x in pr_prec],
        "recall": [round(float(x), 5) for x in pr_rec],
        "average_precision": avg_precision,
    }

    # ── Threshold analysis ────────────────────────────────────────────
    threshold_table = compute_threshold_analysis(y_true, y_proba, thresholds)

    # ── Calibration table ─────────────────────────────────────────────
    calibration = compute_calibration_table(y_true, y_proba, n_bins=10)

    return {
        "model": name,
        "n_test_samples": int(len(y_true)),
        "metrics_at_0.50_threshold": metrics,
        "confusion_matrix": cm,
        "ks_statistic": ks,
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "threshold_analysis": threshold_table,
        "calibration_table": calibration,
    }


# ---------------------------------------------------------------------------
# Cross-validation (training data only)
# ---------------------------------------------------------------------------


def run_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run stratified K-fold cross-validation on the training set only.

    The held-out test set is never touched here.  ``cross_val_score`` clones
    and refits the pipeline for each fold, so imputer/scaler statistics are
    re-derived from each fold's training portion — no intra-CV leakage.

    Parameters
    ----------
    pipeline:
        Unfitted sklearn Pipeline (will be cloned internally for each fold).
    X_train, y_train:
        Training features and labels only.  Test set is excluded.
    cv:
        Number of folds (default 5).
    random_state:
        Seed for ``StratifiedKFold`` shuffling.

    Returns
    -------
    dict
        ``mean_roc_auc``, ``std_roc_auc``, ``fold_scores``.
    """
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=kfold, scoring="roc_auc", n_jobs=-1
    )
    return {
        "cv_folds": cv,
        "fold_scores": [round(float(s), 4) for s in scores],
        "mean_roc_auc": round(float(scores.mean()), 4),
        "std_roc_auc": round(float(scores.std()), 4),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_full_comparison(
    models: dict[str, Pipeline],
    cv_pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the full Phase 2 evaluation for all models on the shared test set.

    Parameters
    ----------
    models:
        ``{display_name: fitted_pipeline}`` dict.
    cv_pipelines:
        ``{display_name: unfitted_pipeline}`` dict for CV (cloned per fold).
    X_train, X_test, y_train, y_test:
        Single shared train/test split.  Both models are evaluated on the
        same X_test / y_test.
    thresholds:
        Decision thresholds for threshold analysis.
    cv_folds:
        Number of CV folds.
    random_state:
        Seed for CV shuffling.

    Returns
    -------
    dict
        Full structured comparison results.
    """
    results: dict[str, Any] = {
        "dataset": "HMEQ Home Equity Loan",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_size_pct": round(len(y_test) / (len(y_train) + len(y_test)) * 100, 1),
        "random_state": random_state,
        "target_column": "bad (renamed to target)",
        "class_balance_test": {
            "bad_0": int((y_test == 0).sum()),
            "bad_1": int((y_test == 1).sum()),
            "bad_1_pct": round(float((y_test == 1).mean() * 100), 2),
        },
        "models": {},
    }

    for display_name, fitted_pipeline in models.items():
        print(f"  Evaluating {display_name} on test set...")
        eval_result = evaluate_model(
            display_name, fitted_pipeline, X_test, y_test, thresholds
        )

        print(f"  Running {cv_folds}-fold CV for {display_name} (training set only)...")
        cv_result = run_cv(
            cv_pipelines[display_name],
            X_train,
            y_train,
            cv=cv_folds,
            random_state=random_state,
        )

        results["models"][display_name] = {**eval_result, "cross_validation": cv_result}

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_roc_pr_curves(comparison: dict[str, Any], output_path: Path) -> None:
    """Generate a 2-panel figure: ROC curve (left) and Precision-Recall curve (right).

    Saves the figure as a PNG file.  Does not display interactively.

    Parameters
    ----------
    comparison:
        Output of ``run_full_comparison()``.
    output_path:
        Destination ``.png`` file path.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in scripts
    import matplotlib.pyplot as plt

    model_data = comparison["models"]
    n_test = comparison["n_test"]
    baseline_prev = comparison["class_balance_test"]["bad_1_pct"] / 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "CreditRisk AI — Model Evaluation\n"
        f"HMEQ Dataset  |  Test set n={n_test}  |  Baseline default rate {baseline_prev:.1%}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    # ── ROC Curve ─────────────────────────────────────────────────────
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    for (name, data), colour in zip(model_data.items(), colours):
        fpr = data["roc_curve"]["fpr"]
        tpr = data["roc_curve"]["tpr"]
        auc = data["metrics_at_0.50_threshold"]["roc_auc"]
        ks = data["ks_statistic"]
        ax_roc.plot(fpr, tpr, lw=2, color=colour,
                    label=f"{name}  AUC={auc:.4f}  KS={ks:.4f}")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10)
    ax_roc.set_title("ROC Curve", fontsize=11, fontweight="bold")
    ax_roc.legend(fontsize=9, loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)

    # ── Precision-Recall Curve ────────────────────────────────────────
    ax_pr = axes[1]
    ax_pr.axhline(baseline_prev, color="k", lw=1, ls="--",
                  label=f"Baseline (default rate {baseline_prev:.1%})")
    for (name, data), colour in zip(model_data.items(), colours):
        prec = data["pr_curve"]["precision"]
        rec = data["pr_curve"]["recall"]
        ap = data["pr_curve"]["average_precision"]
        ax_pr.plot(rec, prec, lw=2, color=colour,
                   label=f"{name}  AP={ap:.4f}")
    ax_pr.set_xlabel("Recall", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax_pr.legend(fontsize=9, loc="upper right")
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves saved → {output_path}")
