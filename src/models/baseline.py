"""
Phase 1 Baseline: Cost-sensitive Logistic Regression with chronological
OOT validation, expanded feature coverage, and artifact serialization.
"""
import logging

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config.settings import (
    ARTIFACTS_DIR,
    DROP_COLS,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET_COL,
)
from src.data.ingestion import DataIngestor
from src.features.engineering import build_feature_pipeline
from src.evaluation.metrics import evaluate_model_on_splits, persist_metrics

logger = logging.getLogger(__name__)


def train_baseline() -> tuple[LogisticRegression, Pipeline, dict[str, dict[str, float]]]:
    """
    End-to-end baseline training pipeline.

    Ingests the SBA National dataset, executes a chronological 70/15/15
    split, fits the full feature preprocessing pipeline on the training
    fold only, trains a cost-sensitive Logistic Regression, evaluates on
    both validation and OOT test sets, and serializes all artifacts.

    Returns
    -------
    tuple[LogisticRegression, Pipeline, dict[str, dict[str, float]]]
        The fitted model, the fitted feature pipeline, and a nested dict
        of evaluation metrics keyed by split name.
    """
    ingestor = DataIngestor(RAW_DATA_PATH)
    df_clean = ingestor.clean()
    train_set, val_set, test_set = ingestor.chronological_split(df=df_clean)

    # Isolate features and target — drop the engineered binary target,
    # the raw MIS_Status string, and ApprovalDate (chronological rank proxy)
    # to prevent leakage.
    y_train = train_set[TARGET_COL].copy()
    X_train = train_set.drop(columns=DROP_COLS).copy()

    y_val = val_set[TARGET_COL].copy()
    X_val = val_set.drop(columns=DROP_COLS).copy()

    y_test = test_set[TARGET_COL].copy()
    X_test = test_set.drop(columns=DROP_COLS).copy()

    # Fit the preprocessing pipeline on the training fold exclusively.
    # Validation and OOT sets receive a stateless .transform() call,
    # ensuring winsorizer bounds and scaler statistics are not leaked.
    pipeline = build_feature_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)

    # class_weight="balanced" assigns inverse-frequency weights to penalise
    # minority-class (default) misclassifications, avoiding SMOTE artefacts.
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_processed, y_train)

    # Evaluate on both held-out splits using the full metric suite
    # (ROC-AUC, PR-AUC, Brier Score, F1/Precision/Recall for default class).
    eval_splits = {"Validation": (X_val_processed, y_val), "OOT Test": (X_test_processed, y_test)}
    metrics = evaluate_model_on_splits(model, eval_splits)
    persist_metrics("baseline", metrics, eval_splits)

    _serialize_artifacts(model, pipeline)

    return model, pipeline, metrics


def _serialize_artifacts(
    model: LogisticRegression,
    pipeline: Pipeline,
) -> None:
    """
    Persist the fitted model and pipeline to disk for reproducibility
    and downstream Phase 2 comparison.

    Artifacts are written to data/processed/ as joblib files, which are
    optimal for large scikit-learn pipelines containing numpy arrays and
    sparse matrices.

    Parameters
    ----------
    model : LogisticRegression
        The fitted baseline classifier.
    pipeline : Pipeline
        The fitted feature preprocessing pipeline.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = ARTIFACTS_DIR / "feature_pipeline.joblib"
    model_path = ARTIFACTS_DIR / "baseline_model.joblib"

    joblib.dump(pipeline, pipeline_path)
    logger.info("Saved feature pipeline → %s", pipeline_path)

    joblib.dump(model, model_path)
    logger.info("Saved baseline model   → %s", model_path)


if __name__ == "__main__":
    from src.utils.logging import configure_logging

    configure_logging()
    train_baseline()
