"""
Phase 2 Gradient Boosting: XGBoost and LightGBM training modules with
early stopping on the chronological validation fold.

Mirrors the data loading and evaluation pattern established in
src/models/baseline.py, substituting the tree-optimised feature pipeline
and gradient boosting classifiers for the Logistic Regression baseline.
"""
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config.settings import (
    ARTIFACTS_DIR,
    DROP_COLS,
    EARLY_STOPPING_ROUNDS,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET_COL,
)
from src.data.ingestion import DataIngestor
from src.evaluation.metrics import evaluate_model_on_splits, persist_metrics
from src.features.engineering import build_tree_feature_pipeline

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
    ingestor = DataIngestor(RAW_DATA_PATH)
    df_clean = ingestor.clean()
    train_set, val_set, test_set = ingestor.chronological_split(df=df_clean)

    y_train = train_set[TARGET_COL].copy()
    X_train = train_set.drop(columns=DROP_COLS).copy()

    y_val = val_set[TARGET_COL].copy()
    X_val = val_set.drop(columns=DROP_COLS).copy()

    y_test = test_set[TARGET_COL].copy()
    X_test = test_set.drop(columns=DROP_COLS).copy()

    tree_pipeline = build_tree_feature_pipeline()
    X_train_processed = tree_pipeline.fit_transform(X_train)
    X_val_processed = tree_pipeline.transform(X_val)
    X_test_processed = tree_pipeline.transform(X_test)

    return (
        X_train_processed, y_train,
        X_val_processed, y_val,
        X_test_processed, y_test,
        tree_pipeline,
    )


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

def train_xgboost(
    prepared: tuple | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[XGBClassifier, Pipeline, dict[str, dict[str, float]]]:
    """
    Train an XGBoost classifier with optional hyperparameters.

    Uses the tree-optimised feature pipeline and early stopping on the
    chronological validation fold. Class imbalance is handled via
    scale_pos_weight computed from the training target distribution.

    Parameters
    ----------
    prepared : tuple or None
        Pre-computed output of _prepare_data(), as returned when calling
        from __main__ alongside train_lightgbm(). If None, this function
        calls _prepare_data() itself (e.g. when called standalone from
        the tuning module).
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
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        tree_pipeline,
    ) = prepared if prepared is not None else _prepare_data()

    counts = y_train.value_counts().reindex([0, 1], fill_value=0)
    scale_pos_weight = counts[0] / counts[1]

    model_params = {
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
    }
    if params is not None:
        model_params.update(params)

    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    eval_splits = {"Validation": (X_val, y_val), "OOT Test": (X_test, y_test)}
    metrics = evaluate_model_on_splits(model, eval_splits)
    persist_metrics("xgb_model", metrics, eval_splits)
    _serialize_model(model, "xgb_model")
    return model, tree_pipeline, metrics


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lightgbm(
    prepared: tuple | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[LGBMClassifier, Pipeline, dict[str, dict[str, float]]]:
    """
    Train a LightGBM classifier with optional hyperparameters.

    Uses the tree-optimised feature pipeline and early stopping on the
    chronological validation fold. Class imbalance is handled via
    the is_unbalance flag.

    Parameters
    ----------
    prepared : tuple or None
        Pre-computed output of _prepare_data(), as returned when calling
        from __main__ alongside train_xgboost(). If None, this function
        calls _prepare_data() itself (e.g. when called standalone from
        the tuning module).
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
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        tree_pipeline,
    ) = prepared if prepared is not None else _prepare_data()

    model_params = {
        "is_unbalance": True,
        "random_state": RANDOM_SEED,
        "verbosity": -1,
    }
    if params is not None:
        model_params.update(params)

    model = LGBMClassifier(**model_params)
    # eval_set alone only records history — early_stopping must be passed
    # as a callback or LightGBM trains for the full n_estimators regardless
    # of validation performance.
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS)],
    )

    eval_splits = {"Validation": (X_val, y_val), "OOT Test": (X_test, y_test)}
    metrics = evaluate_model_on_splits(model, eval_splits)
    persist_metrics("lgb_model", metrics, eval_splits)
    _serialize_model(model, "lgb_model")
    return model, tree_pipeline, metrics


# ---------------------------------------------------------------------------
# Artifact serialization
# ---------------------------------------------------------------------------

def _serialize_model(model: XGBClassifier | LGBMClassifier, model_name: str) -> None:
    """
    Persist a fitted model to disk.

    Parameters
    ----------
    model : XGBClassifier or LGBMClassifier
        The fitted gradient boosting classifier.
    model_name : str
        Base name for the serialized file (e.g., "xgb_model", "lgb_model").
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved %s model → %s", model_name, model_path)


def _serialize_pipeline(pipeline: Pipeline) -> None:
    """
    Persist the shared tree feature pipeline to disk.

    Both XGBoost and LightGBM are trained on the same tree-optimised
    pipeline, so this is called once per __main__ run rather than once
    per model — a second call would just overwrite the first with an
    identical pipeline fitted on the same training fold.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted tree feature preprocessing pipeline.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = ARTIFACTS_DIR / "tree_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info("Saved tree feature pipeline → %s", pipeline_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.logging import configure_logging

    configure_logging()

    prepared = _prepare_data()
    _serialize_pipeline(prepared[-1])
    train_xgboost(prepared=prepared)
    train_lightgbm(prepared=prepared)
