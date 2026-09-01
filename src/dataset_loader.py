"""
HMEQ dataset loader for CreditRisk AI.

This module is the single entry point for loading the HMEQ Home Equity
Loan dataset.  It validates the file, the expected columns, and the
target variable before returning a clean DataFrame.

Behaviour contract
------------------
* Raises ``FileNotFoundError`` with clear instructions when the CSV is
  absent — it never silently generates synthetic replacement data.
* Raises ``ValueError`` with a precise message when expected columns are
  missing or the target column contains unexpected values.
* All column names are lowercased internally so the rest of the codebase
  uses consistent snake_case names.

Dataset reference
-----------------
Source  : SAS Institute / SAS Viya DMML Pipelines
URL     : https://raw.githubusercontent.com/sassoftware/
          sas-viya-dmml-pipelines/master/data/hmeq.csv
Rows    : 5,960
Target  : BAD  →  1 = bad/defaulted loan ; 0 = non-bad loan
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Expected schema (uppercase as found in the raw CSV; lowercased internally)
# ---------------------------------------------------------------------------

HMEQ_EXPECTED_COLUMNS: tuple[str, ...] = (
    "bad",       # Target: 1 = bad/defaulted loan, 0 = non-bad loan
    "loan",      # Amount of the loan request ($)
    "mortdue",   # Amount due on the existing mortgage ($)
    "value",     # Current property value ($)
    "reason",    # Loan purpose: DebtCon | HomeImp
    "job",       # Occupational category (6 levels)
    "yoj",       # Years at present job
    "derog",     # Number of major derogatory reports
    "delinq",    # Number of delinquent credit lines
    "clage",     # Age of oldest credit line (months)
    "ninq",      # Number of recent credit inquiries
    "clno",      # Number of credit lines
    "debtinc",   # Debt-to-income ratio (%)
)

HMEQ_TARGET_COLUMN: str = "bad"
HMEQ_EXPECTED_TARGET_VALUES: frozenset[int] = frozenset({0, 1})

DOWNLOAD_INSTRUCTIONS: str = (
    "\n"
    "  To obtain the HMEQ dataset:\n"
    "  Option 1 (direct download, no login required):\n"
    "    curl -o data/raw/hmeq.csv \\\n"
    "      https://raw.githubusercontent.com/sassoftware/"
    "sas-viya-dmml-pipelines/master/data/hmeq.csv\n"
    "\n"
    "  Option 2 (Kaggle):\n"
    "    https://www.kaggle.com/datasets/ajaymanwani/hmeq-data\n"
    "    Download and place the CSV at:  data/raw/hmeq.csv\n"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_hmeq(path: str | Path) -> pd.DataFrame:
    """Load and validate the HMEQ CSV dataset.

    Parameters
    ----------
    path:
        Absolute or relative path to ``hmeq.csv``.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with lowercased column names.

    Raises
    ------
    FileNotFoundError
        When the CSV file does not exist at ``path``.
    ValueError
        When expected columns are missing from the file.
    ValueError
        When the target column contains values outside {0, 1}.
    """
    path = Path(path)

    # ---- file existence -------------------------------------------------
    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found at: {path}\n"
            f"{DOWNLOAD_INSTRUCTIONS}"
        )

    # ---- read CSV -------------------------------------------------------
    raw = pd.read_csv(path, low_memory=False)

    # ---- normalise column names (lowercase + strip) ---------------------
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    # ---- column presence validation ------------------------------------
    missing_columns = [c for c in HMEQ_EXPECTED_COLUMNS if c not in raw.columns]
    if missing_columns:
        raise ValueError(
            f"\nThe following expected HMEQ columns are missing from the CSV:\n"
            f"  {missing_columns}\n"
            f"\nColumns found in file: {raw.columns.tolist()}\n"
            f"\nEnsure you are using the standard HMEQ dataset.{DOWNLOAD_INSTRUCTIONS}"
        )

    # ---- target column validation --------------------------------------
    _validate_target(raw)

    return raw


def _validate_target(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the target column is absent or has unexpected values."""
    if HMEQ_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{HMEQ_TARGET_COLUMN}' not found in the dataset."
        )

    target = df[HMEQ_TARGET_COLUMN].dropna()

    if len(target) == 0:
        raise ValueError(
            f"Target column '{HMEQ_TARGET_COLUMN}' contains only missing values."
        )

    observed_values = set(target.astype(int).unique())
    unexpected = observed_values - HMEQ_EXPECTED_TARGET_VALUES
    if unexpected:
        raise ValueError(
            f"Target column '{HMEQ_TARGET_COLUMN}' contains unexpected values: "
            f"{unexpected}.  Expected only {{0, 1}}."
        )

    n_classes = len(observed_values)
    if n_classes < 2:
        raise ValueError(
            f"Target column '{HMEQ_TARGET_COLUMN}' has only one unique value "
            f"({observed_values}).  At least two classes are required for "
            f"binary classification."
        )


def generate_data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Produce a structured data quality report for the loaded DataFrame.

    Parameters
    ----------
    df:
        DataFrame as returned by :func:`load_hmeq`.

    Returns
    -------
    dict
        Structured report covering shape, column types, missing values,
        duplicates, and target distribution.
    """
    n_rows, n_cols = df.shape

    # ---- missing values per column ------------------------------------
    missing_counts = df.isnull().sum()
    missing_report: list[dict[str, Any]] = []
    for col in df.columns:
        count = int(missing_counts[col])
        pct = round(count / n_rows * 100, 2) if n_rows > 0 else 0.0
        missing_report.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": count,
                "missing_pct": pct,
            }
        )

    # ---- duplicate rows -----------------------------------------------
    duplicate_count = int(df.duplicated().sum())

    # ---- target distribution (BAD) ------------------------------------
    target_report: dict[str, Any] = {}
    if HMEQ_TARGET_COLUMN in df.columns:
        counts = df[HMEQ_TARGET_COLUMN].value_counts(dropna=False).sort_index()
        bad_0 = int(counts.get(0, 0))
        bad_1 = int(counts.get(1, 0))
        total_non_null = bad_0 + bad_1
        target_report = {
            "target_column": HMEQ_TARGET_COLUMN,
            "bad_0_non_bad_loan": bad_0,
            "bad_1_defaulted_loan": bad_1,
            "bad_1_pct": round(bad_1 / total_non_null * 100, 2) if total_non_null else 0.0,
            "missing_target": int(df[HMEQ_TARGET_COLUMN].isnull().sum()),
        }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": df.columns.tolist(),
        "missing_values": missing_report,
        "duplicate_rows": duplicate_count,
        "target_distribution": target_report,
    }
