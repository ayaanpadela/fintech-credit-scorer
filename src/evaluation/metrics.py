"""
Reusable evaluation metrics module for credit scoring models.

Computes the full regulatory-grade metric suite — ROC-AUC, PR-AUC,
Brier Score, and per-class classification statistics — across arbitrary
data splits. Designed to be consumed by both the Phase 1 baseline and
Phase 2 gradient boosting training scripts.
"""
import hashlib
import inspect
import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)

import config.settings as settings
from config.settings import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

_TRACKED_LIBRARIES = ("scikit-learn", "xgboost", "lightgbm", "pandas", "numpy")


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split_name: str = "Test",
) -> dict[str, float]:
    """
    Compute and log the full evaluation suite for a single data split.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels.
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (from threshold).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive (default) class.
    split_name : str
        Human-readable name for logging (e.g., "Validation", "OOT Test").

    Returns
    -------
    dict[str, float]
        Dictionary containing: roc_auc, pr_auc, brier_score, f1_default,
        precision_default, recall_default.
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    report = classification_report(y_true, y_pred, output_dict=True)
    f1_default = report.get("1", {}).get("f1-score", float("nan"))
    precision_default = report.get("1", {}).get("precision", float("nan"))
    recall_default = report.get("1", {}).get("recall", float("nan"))

    logger.info(
        "[%s] ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f | "
        "F1-Default: %.4f | Precision-Default: %.4f | Recall-Default: %.4f",
        split_name,
        roc_auc,
        pr_auc,
        brier,
        f1_default,
        precision_default,
        recall_default,
    )
    logger.info(
        "[%s] Full Classification Report:\n%s",
        split_name,
        classification_report(y_true, y_pred),
    )

    # Explicit float() cast — sklearn returns np.floating which
    # serialises inconsistently with json.dumps.
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "f1_default": float(f1_default),
        "precision_default": float(precision_default),
        "recall_default": float(recall_default),
    }


def evaluate_model_on_splits(
    model: Any,
    splits: dict[str, tuple[np.ndarray, np.ndarray | pd.Series]],
) -> dict[str, dict[str, float]]:
    """
    Run the full evaluation suite across multiple named splits.

    Parameters
    ----------
    model : Any
        A fitted scikit-learn compatible classifier exposing .predict()
        and .predict_proba() methods.
    splits : dict[str, tuple[np.ndarray, np.ndarray | pd.Series]]
        Mapping of split_name -> (X_processed, y_true).
        Example: {"Validation": (X_val, y_val), "OOT Test": (X_test, y_test)}

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict of split_name -> metrics_dict.
    """
    all_results: dict[str, dict[str, float]] = {}

    for split_name, (X_processed, y_true) in splits.items():
        y_pred = model.predict(X_processed)
        y_prob = model.predict_proba(X_processed)[:, 1]

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            split_name=split_name,
        )
        all_results[split_name] = metrics

    return all_results


def _config_hash() -> str:
    """
    Hash the source of config/settings.py so persisted metrics can be
    traced back to the exact column registries, split ratios, and seeds
    that produced them. Changes to any constant change the hash.
    """
    source_path = Path(inspect.getsourcefile(settings))
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return digest[:12]


def _library_versions() -> dict[str, str]:
    versions = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = pkg_version(name)
        except Exception:
            versions[name] = "unknown"
    return versions


def persist_metrics(
    model_name: str,
    metrics: dict[str, dict[str, float]],
    splits: dict[str, tuple[np.ndarray, np.ndarray | pd.Series]],
) -> Path:
    """
    Write evaluation metrics to data/processed/metrics_{model_name}.json
    so README tables can be regenerated from a persisted audit trail
    rather than hand-transcribed.

    Parameters
    ----------
    model_name : str
        Identifies the model in the output filename (e.g. "baseline",
        "xgb_model", "lgb_model").
    metrics : dict[str, dict[str, float]]
        Output of evaluate_model_on_splits() — split_name -> metrics_dict.
    splits : dict[str, tuple[np.ndarray, np.ndarray | pd.Series]]
        The same splits mapping passed to evaluate_model_on_splits(), used
        here only to record row counts per split.

    Returns
    -------
    Path
        The path the metrics were written to.
    """
    payload = {
        "model": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": _config_hash(),
        "library_versions": _library_versions(),
        "split_row_counts": {name: len(y) for name, (_, y) in splits.items()},
        "metrics": metrics,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / f"metrics_{model_name}.json"
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved metrics → %s", output_path)
    return output_path