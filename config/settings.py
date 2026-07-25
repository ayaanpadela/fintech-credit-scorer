"""
Centralized configuration constants for the FinTech Credit Scorer.

This module is the single source of truth for all paths, column lists,
random seeds, and hyperparameter search spaces consumed by the
data ingestion, feature engineering, and model training modules.

Constraint: This module contains ONLY constants. No functions, no classes,
no imports from src/.
"""
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "SBAnational.csv"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

CENTURY_CUTOFF_YEAR: int = 2026
TARGET_COL: str = "is_default"
DROP_COLS: list[str] = ["is_default", "MIS_Status", "ApprovalDate"]
SPLIT_RATIOS: tuple[float, float] = (0.70, 0.85)
RANDOM_SEED: int = 42

# Columns that are only known once a loan's outcome is already determined.
# Including any of these as model features would leak the target. Dropped
# unconditionally in DataIngestor.clean(), independent of which features
# the downstream pipeline happens to select.
POST_OUTCOME_COLS: list[str] = [
    "ChgOffDate",
    "ChgOffPrinGr",
    "BalanceGross",
    "DisbursementGross",
    "DisbursementDate",
]

# Sanity band for the engineered target's positive rate. If a schema or
# encoding change silently alters how "CHGOFF" is represented in
# MIS_Status, the resulting positive rate should fail loudly rather than
# silently train a model on a corrupted target.
POSITIVE_RATE_BOUNDS: tuple[float, float] = (0.05, 0.40)

# ---------------------------------------------------------------------------
# Feature column registries
# ---------------------------------------------------------------------------

FINANCIAL_COLS: list[str] = [
    "GrAppv",
    "SBA_Appv",
]

CONTINUOUS_FEATURES: list[str] = [
    "GrAppv",
    "SBA_Appv",
    "GOV_Ratio",
    "Term",
    "NoEmp",
    "CreateJob",
    "RetainedJob",
]

CATEGORICAL_FEATURES: list[str] = [
    "NAICS",
    "RevLineCr",
    "LowDoc",
    "NewExist",
    "UrbanRural"
]

PASSTHROUGH_BINARY: list[str] = [
    "is_backed",
    "is_franchise",
]

WINSORIZE_COLS: list[str] = [
    "GrAppv",
    "SBA_Appv",
    "Term",
    "NoEmp",
    "CreateJob",
    "RetainedJob",
]

# Allow-list mapping consumed by CategoricalSanitizer: column -> {raw
# value: canonical value}. Values absent from a column's map are filled
# with "Unknown" by the transformer.
CATEGORICAL_VALUE_MAP: dict[str, dict[str, str]] = {
    "RevLineCr": {
        "Y": "Y", "N": "N",
        "y": "Y", "n": "N",
        "0": "Unknown", "T": "Unknown",
        "nan": "Unknown",
    },
    "LowDoc": {
        "Y": "Y", "N": "N",
        "y": "Y", "n": "N",
        "S": "Unknown", "C": "Unknown",
        "0": "Unknown",
        "nan": "Unknown",
    },
    "NewExist": {
        "1": "Existing", "1.0": "Existing",
        "2": "New",      "2.0": "New",
        "0": "Unknown",
        "nan": "Unknown",
    },
}

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

N_OPTUNA_TRIALS: int = 50
EARLY_STOPPING_ROUNDS: int = 50

# XGBoost search space — dict values are (low, high) tuples.
# The tuning module interprets the key suffix to select the Optuna method:
#   _int   → trial.suggest_int
#   _float → trial.suggest_float
#   _log   → trial.suggest_float(log=True)
XGB_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "max_depth_int": (3, 10),
    "learning_rate_log": (0.01, 0.3),
    "n_estimators_int": (100, 1000),
    "subsample_float": (0.6, 1.0),
    "colsample_bytree_float": (0.6, 1.0),
    "min_child_weight_int": (1, 10),
    "reg_alpha_log": (1e-8, 10.0),
    "reg_lambda_log": (1e-8, 10.0),
    "gamma_log": (1e-8, 5.0),
}

# LightGBM search space — same suffix convention as XGBoost.
LGB_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "max_depth_int": (3, 12),
    "learning_rate_log": (0.01, 0.3),
    "n_estimators_int": (100, 1000),
    "num_leaves_int": (20, 300),
    "subsample_float": (0.6, 1.0),
    "colsample_bytree_float": (0.6, 1.0),
    "min_child_samples_int": (5, 100),
    "reg_alpha_log": (1e-8, 10.0),
    "reg_lambda_log": (1e-8, 10.0),
}