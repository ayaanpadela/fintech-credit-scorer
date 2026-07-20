"""
Reusable evaluation metrics module for credit scoring models.

Computes the full regulatory-grade metric suite — ROC-AUC, PR-AUC,
Brier Score, and per-class classification statistics — across arbitrary
data splits. Designed to be consumed by both the Phase 1 baseline and
Phase 2 gradient boosting training scripts.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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