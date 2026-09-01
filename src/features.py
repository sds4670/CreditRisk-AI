"""
Feature engineering for the HMEQ Home Equity Loan dataset.

All features are derived exclusively from columns confirmed to exist in
the HMEQ dataset: loan, mortdue, value, derog, delinq, clno, clage.

Safe ratio rules
----------------
* Division uses ``np.where(denominator > 0, numerator / denominator, np.nan)``
  so zero denominators are replaced with NaN, not infinity.
* After division, an explicit ``replace([np.inf, -np.inf], np.nan)`` pass
  ensures no infinite values survive (guards against edge-case float
  arithmetic on very small denominators).
* Downstream sklearn ``SimpleImputer(strategy='median')`` in the
  preprocessing pipeline handles remaining NaN values.

Design principle
----------------
``engineer_features`` is a pure function: it receives a DataFrame and
returns a new enriched DataFrame.  It does NOT fit any statistics — all
parameters (thresholds, scales) are hard-coded constants derived from
domain knowledge, not from the data.  This prevents data leakage.

Engineered features summary
---------------------------
Feature              Formula                          Business rationale
-----------          ---------------------------      ----------------------------------------
loan_to_value        LOAN / VALUE                     LTV ratio — core underwriting metric;
                                                      higher LTV = less collateral cushion
mort_to_value        MORTDUE / VALUE                  Existing mortgage burden vs property
                                                      value; indicates equity already pledged
total_debt_value     (LOAN + MORTDUE) / VALUE         Combined leverage ratio; proxy for
                                                      total equity erosion
has_derog            (DEROG > 0) → binary int         Any major derogatory report is a strong
                                                      standalone default signal
has_delinq           (DELINQ > 0) → binary int        Any delinquent credit line flags active
                                                      payment distress
delinq_severity      DELINQ / CLNO                   Fraction of credit lines that are
                                                      delinquent; normalises raw delinquency
                                                      count by credit portfolio size
clage_years          CLAGE / 12                       Credit history length in years
                                                      (interpretability; same information
                                                      content as clage in months)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns required for engineering (lowercased, as produced by dataset_loader)
_REQUIRED_FOR_RATIOS: tuple[str, ...] = ("loan", "mortdue", "value")
_REQUIRED_FOR_FLAGS: tuple[str, ...] = ("derog", "delinq")
_REQUIRED_FOR_SEVERITY: tuple[str, ...] = ("delinq", "clno")
_REQUIRED_FOR_CLAGE: tuple[str, ...] = ("clage",)

ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "loan_to_value",
    "mort_to_value",
    "total_debt_value",
    "has_derog",
    "has_delinq",
    "delinq_severity",
    "clage_years",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_divide(
    numerator: pd.Series, denominator: pd.Series
) -> pd.Series:
    """Divide two Series element-wise, replacing invalid results with NaN.

    Zero denominators → NaN (avoids ZeroDivisionError and inf).
    Infinite results → NaN (guards against float edge-cases on tiny
    denominators that are > 0 but very close to zero).
    Missing values in either operand propagate naturally as NaN.
    """
    num_arr = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den_arr = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)

    # np.where evaluates both branches before masking, so division by zero
    # and invalid-value warnings are expected and safe to suppress here.
    # The mask (den_arr > 0) ensures zero-denominator positions receive NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(den_arr > 0, num_arr / den_arr, np.nan)
    result = np.where(np.isfinite(result), result, np.nan)
    return pd.Series(result, index=numerator.index, dtype=float)



def _columns_present(df: pd.DataFrame, required: tuple[str, ...]) -> bool:
    """Return True only when all required columns are present in df."""
    return all(c in df.columns for c in required)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features for the HMEQ dataset.

    This function is a pure transformation: it does not fit any statistics
    on the data.  Call it *after* the train/test split if you need to
    ensure there is absolutely no leakage, or on the full DataFrame before
    splitting (safe here because all transformations use fixed constants).

    Parameters
    ----------
    df:
        DataFrame as returned by ``load_hmeq()``.  Column names must be
        lowercase (the loader guarantees this).

    Returns
    -------
    pd.DataFrame
        New DataFrame containing all original columns plus the engineered
        features listed in ``ENGINEERED_FEATURE_NAMES``.  The original
        DataFrame is not modified.
    """
    frame = df.copy()
    created: list[str] = []

    # ------------------------------------------------------------------ #
    # Ratio features — all guarded with _safe_divide                      #
    # ------------------------------------------------------------------ #

    if _columns_present(frame, _REQUIRED_FOR_RATIOS):
        loan = pd.to_numeric(frame["loan"], errors="coerce")
        mortdue = pd.to_numeric(frame["mortdue"], errors="coerce")
        value = pd.to_numeric(frame["value"], errors="coerce")

        # LOAN_TO_VALUE: loan request as share of property value
        frame["loan_to_value"] = _safe_divide(loan, value)
        created.append("loan_to_value")

        # MORT_TO_VALUE: existing mortgage as share of property value
        frame["mort_to_value"] = _safe_divide(mortdue, value)
        created.append("mort_to_value")

        # TOTAL_DEBT_VALUE: combined debt (loan + mortgage) vs property value
        frame["total_debt_value"] = _safe_divide(loan + mortdue, value)
        created.append("total_debt_value")

    # ------------------------------------------------------------------ #
    # Binary derogatory / delinquency flags                               #
    # ------------------------------------------------------------------ #

    if _columns_present(frame, _REQUIRED_FOR_FLAGS):
        derog = pd.to_numeric(frame["derog"], errors="coerce")
        delinq = pd.to_numeric(frame["delinq"], errors="coerce")

        # HAS_DEROG: 1 if any major derogatory report exists, else 0
        # NaN derog → treated as 0 (no known derogatory record) for the flag;
        # the original derog column retains its NaN for the imputer.
        frame["has_derog"] = (derog.fillna(0) > 0).astype(int)
        created.append("has_derog")

        # HAS_DELINQ: 1 if any delinquent credit line exists, else 0
        frame["has_delinq"] = (delinq.fillna(0) > 0).astype(int)
        created.append("has_delinq")

    # ------------------------------------------------------------------ #
    # DELINQ_SEVERITY: fraction of credit lines that are delinquent       #
    # ------------------------------------------------------------------ #

    if _columns_present(frame, _REQUIRED_FOR_SEVERITY):
        delinq = pd.to_numeric(frame["delinq"], errors="coerce")
        clno = pd.to_numeric(frame["clno"], errors="coerce")

        frame["delinq_severity"] = _safe_divide(delinq, clno)
        created.append("delinq_severity")

    # ------------------------------------------------------------------ #
    # CLAGE_YEARS: credit history age converted from months to years      #
    # ------------------------------------------------------------------ #

    if _columns_present(frame, _REQUIRED_FOR_CLAGE):
        clage = pd.to_numeric(frame["clage"], errors="coerce")
        frame["clage_years"] = clage / 12.0
        created.append("clage_years")

    return frame


def get_feature_metadata() -> list[dict[str, str]]:
    """Return a list of dicts documenting each engineered feature.

    Useful for generating reports and README documentation.
    """
    return [
        {
            "name": "loan_to_value",
            "formula": "LOAN / VALUE",
            "rationale": "Loan-to-value ratio; higher LTV = less collateral cushion.",
            "safe_handling": "Zero or missing VALUE → NaN; imputed downstream.",
        },
        {
            "name": "mort_to_value",
            "formula": "MORTDUE / VALUE",
            "rationale": "Existing mortgage burden relative to property value.",
            "safe_handling": "Zero or missing VALUE → NaN; imputed downstream.",
        },
        {
            "name": "total_debt_value",
            "formula": "(LOAN + MORTDUE) / VALUE",
            "rationale": "Combined leverage ratio; proxy for total equity erosion.",
            "safe_handling": "Zero or missing VALUE → NaN; imputed downstream.",
        },
        {
            "name": "has_derog",
            "formula": "(DEROG > 0).astype(int)",
            "rationale": "Binary flag; any major derogatory record is a strong default signal.",
            "safe_handling": "Missing DEROG treated as 0 for flag only; original column unchanged.",
        },
        {
            "name": "has_delinq",
            "formula": "(DELINQ > 0).astype(int)",
            "rationale": "Binary flag; any delinquent credit line flags active payment distress.",
            "safe_handling": "Missing DELINQ treated as 0 for flag only; original column unchanged.",
        },
        {
            "name": "delinq_severity",
            "formula": "DELINQ / CLNO",
            "rationale": "Fraction of credit lines delinquent; normalises raw count by portfolio size.",
            "safe_handling": "Zero or missing CLNO → NaN; imputed downstream.",
        },
        {
            "name": "clage_years",
            "formula": "CLAGE / 12",
            "rationale": "Credit history length in years for interpretability.",
            "safe_handling": "Missing CLAGE → NaN propagates; imputed downstream.",
        },
    ]
