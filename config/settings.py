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

TARGET_COL: str = "is_default"
DROP_COLS: list[str] = ["is_default", "MIS_Status","ApprovalDate"]
SPLIT_RATIOS: tuple[float, float] = (0.70, 0.85)
RANDOM_SEED: int = 42

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
]

PASSTHROUGH_BINARY: list[str] = [
    "is_backed",
    "UrbanRural",
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