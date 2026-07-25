"""
Phase 2 Gradient Boosting: XGBoost and LightGBM training modules with
early stopping on the chronological validation fold.

Mirrors the data loading and evaluation pattern established in
src/models/baseline.py, substituting the tree-optimised feature pipeline
and gradient boosting classifiers for the Logistic Regression baseline.
"""
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config.settings import (
    ARTIFACTS_DIR,
    DROP_COLS,
    EARLY_STOPPING_ROUNDS,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    RANDOM_SEED,
    TARGET_COL,
)
from src.data.ingestion import DataIngestor
from src.features.engineering import build_tree_feature_pipeline
from src.evaluation.metrics import evaluate_model_on_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation helper
# ---------------------------------------------------------------------------

def _prepare_data() -> tuple[
    np.ndarray, pd.Series,
    np.ndarray, pd.Series,
    np.ndarray, pd.Series,
    Pipeline,
]:
    """
    Load, split, and preprocess the SBA dataset for tree model training.

    Executes the full data loading → chronological split → tree pipeline
    fit_transform flow. The pipeline is fitted exclusively on the training
    fold to prevent data leakage.

    Returns
    -------
    tuple containing:
        X_train_processed : np.ndarray
        y_train : pd.Series
        X_val_processed : np.ndarray
        y_val : pd.Series
        X_test_processed : np.ndarray
        y_test : pd.Series
        pipeline : Pipeline (fitted)
    """
    # TODO: Implement data loading, splitting, and pipeline fitting.
    # Follow the same pattern as baseline.py:
    #   1. DataIngestor(RAW_DATA_PATH).clean()
    #   2. .chronological_split(df_clean)
    #   3. Isolate X/y using TARGET_COL and DROP_COLS
    #   4. build_tree_feature_pipeline().fit_transform(X_train)
    #   5. pipeline.transform(X_val), pipeline.transform(X_test)
    ingestor = DataIngestor(RAW_DATA_PATH)
    df_clean = ingestor.clean()
    train_set, val_set, test_set = ingestor.chronological_split(df=df_clean)
    y_train = train_set[TARGET_COL].copy()
    X_train = train_set.drop(columns=DROP_COLS).copy()

    y_val = val_set[TARGET_COL].copy()
    X_val = val_set.drop(columns=DROP_COLS).copy()

    y_test = test_set[TARGET_COL].copy()
    X_test = test_set.drop(columns=DROP_COLS).copy()
    tree_pipeline= build_tree_feature_pipeline()
    X_train_processed=tree_pipeline.fit_transform(X_train)
    X_val_processed= tree_pipeline.transform(X_val)
    X_test_processed= tree_pipeline.transform(X_test)

    return (X_train_processed,y_train,X_val_processed,y_val,X_test_processed,y_test,tree_pipeline)


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(
    params: dict[str, Any] | None = None,
) -> tuple[XGBClassifier, Pipeline, dict[str, dict[str, float]]]:
    """
    Train an XGBoost classifier with optional hyperparameters.

    Uses the tree-optimised feature pipeline and early stopping on the
    chronological validation fold. Class imbalance is handled via
    scale_pos_weight computed from the training target distribution.

    Parameters
    ----------
    params : dict[str, Any] or None
        XGBoost hyperparameters. If None, uses sensible defaults.
        When called from the tuning module, this contains the
        Optuna-suggested parameters.

    Returns
    -------
    tuple[XGBClassifier, Pipeline, dict[str, dict[str, float]]]
        The fitted model, fitted pipeline, and evaluation metrics
        keyed by split name ("Validation", "OOT Test").
    """
    # TODO: Implement XGBoost training.
    # Steps:
    #   1. Call _prepare_data() to get processed matrices
    #   2. Compute scale_pos_weight = count(y=0) / count(y=1)
    #   3. Merge default params with any provided params
    #   4. Instantiate XGBClassifier with:
    #      - scale_pos_weight
    #      - early_stopping_rounds=EARLY_STOPPING_ROUNDS
    #      - eval_metric="logloss"
    #      - random_state=RANDOM_SEED
    #      - tree_method="hist"
    #   5. model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    #   6. Evaluate using evaluate_model_on_splits()
    #   7. Serialize artifacts
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        tree_pipeline
    ) = _prepare_data()
    counts = y_train.value_counts().reindex([0, 1], fill_value=0)
    scale_pos_weight = counts[0]/counts[1]
    model_params = {
    "scale_pos_weight": scale_pos_weight,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "tree_method": "hist"
    }

    if params is not None:
        model_params.update(params)
    xgbmodel= XGBClassifier(**model_params)
    xgbmodel.fit(X_train,y_train,eval_set=[(X_val,y_val)])
    metrics=evaluate_model_on_splits(xgbmodel,{"Validation":(X_val,y_val),"OOT Test":(X_test,y_test)})
    _serialize_artifacts(xgbmodel,tree_pipeline,"xgb_model")
    return (xgbmodel,tree_pipeline,metrics)


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lightgbm(
    params: dict[str, Any] | None = None,
) -> tuple[LGBMClassifier, Pipeline, dict[str, dict[str, float]]]:
    """
    Train a LightGBM classifier with optional hyperparameters.

    Uses the tree-optimised feature pipeline and early stopping on the
    chronological validation fold. Class imbalance is handled via
    the is_unbalance flag.

    Parameters
    ----------
    params : dict[str, Any] or None
        LightGBM hyperparameters. If None, uses sensible defaults.
        When called from the tuning module, this contains the
        Optuna-suggested parameters.

    Returns
    -------
    tuple[LGBMClassifier, Pipeline, dict[str, dict[str, float]]]
        The fitted model, fitted pipeline, and evaluation metrics
        keyed by split name ("Validation", "OOT Test").
    """
    # TODO: Implement LightGBM training.
    # Steps:
    #   1. Call _prepare_data() to get processed matrices
    #   2. Merge default params with any provided params
    #   3. Instantiate LGBMClassifier with:
    #      - is_unbalance=True
    #      - random_state=RANDOM_SEED
    #      - verbosity=-1
    #   4. model.fit(X_train, y_train,
    #                eval_set=[(X_val, y_val)],
    #                callbacks=[early_stopping(EARLY_STOPPING_ROUNDS)])
    #   5. Evaluate using evaluate_model_on_splits()
    #   6. Serialize artifacts
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        tree_pipeline
    ) = _prepare_data()
    model_params={
        "is_unbalance":True,
        "random_state": RANDOM_SEED,
        "verbosity":-1
    }
    if params is not None:
        model_params.update(params)
    lightgbm= LGBMClassifier(**model_params)
    lightgbm.fit(X_train,y_train,eval_set=[(X_val,y_val)])
    metrics=evaluate_model_on_splits(lightgbm,{"Validation":(X_val,y_val),"OOT Test":(X_test,y_test)})
    _serialize_artifacts(lightgbm,tree_pipeline,"lgb_model")
    return (lightgbm,tree_pipeline,metrics)

# ---------------------------------------------------------------------------
# Artifact serialization
# ---------------------------------------------------------------------------

def _serialize_artifacts(
    model: XGBClassifier | LGBMClassifier,
    pipeline: Pipeline,
    model_name: str,
) -> None:
    """
    Persist the fitted model and tree pipeline to disk.

    Parameters
    ----------
    model : XGBClassifier or LGBMClassifier
        The fitted gradient boosting classifier.
    pipeline : Pipeline
        The fitted tree feature preprocessing pipeline.
    model_name : str
        Base name for the serialized file (e.g., "xgb_model", "lgb_model").
    """
    # TODO: Implement serialization.
    # Steps:
    #   1. ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    #   2. joblib.dump(model, ARTIFACTS_DIR / f"{model_name}.joblib")
    #   3. joblib.dump(pipeline, ARTIFACTS_DIR / "tree_pipeline.joblib")
    #   4. Log saved paths
    ARTIFACTS_DIR.mkdir(parents=True,exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR/"tree_pipeline.joblib")
    logger.info("Saved feature pipeline → %s",ARTIFACTS_DIR/"tree_pipeline.joblib")

    joblib.dump(model, ARTIFACTS_DIR/f"{model_name}.joblib")
    logger.info("Saved baseline model   → %s", ARTIFACTS_DIR/f"{model_name}.joblib")
    pass
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_xgboost()
    train_lightgbm()