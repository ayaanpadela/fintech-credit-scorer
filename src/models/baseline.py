"""
Phase 1 Baseline: Cost-sensitive Logistic Regression with chronological
OOT validation, expanded feature coverage, and artifact serialization.
"""
import logging
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.ingestion import DataIngestor
from src.features.engineering import build_feature_pipeline
from src.evaluation.metrics import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
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
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "raw" / "SBAnational.csv"

    ingestor = DataIngestor(data_dir)
    df_clean = ingestor.clean()
    train_set, val_set, test_set = ingestor.chronological_split(df=df_clean)

    # Isolate features and target — drop the engineered binary target,
    # the raw MIS_Status string, and ApprovalDate (chronological rank proxy)
    # to prevent leakage. Mirrors DROP_COLS in config/settings.py.
    drop_cols = ["is_default", "MIS_Status", "ApprovalDate"]

    y_train = train_set["is_default"].copy()
    X_train = train_set.drop(columns=drop_cols).copy()

    y_val = val_set["is_default"].copy()
    X_val = val_set.drop(columns=drop_cols).copy()

    y_test = test_set["is_default"].copy()
    X_test = test_set.drop(columns=drop_cols).copy()

    # Fit the preprocessing pipeline on the training fold exclusively.
    # Validation and OOT sets receive a stateless .transform() call,
    # ensuring winsorizer bounds and scaler statistics are not leaked.
    pipeline = build_feature_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)

    # class_weight="balanced" assigns inverse-frequency weights to penalise
    # minority-class (default) misclassifications, avoiding SMOTE artefacts.
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_processed, y_train)

    # Evaluate on both held-out splits using the full metric suite
    # (ROC-AUC, PR-AUC, Brier Score, F1/Precision/Recall for default class).
    y_val_pred = model.predict(X_val_processed)
    y_val_prob = model.predict_proba(X_val_processed)[:, 1]
    val_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_prob, split_name="Validation")

    y_test_pred = model.predict(X_test_processed)
    y_test_prob = model.predict_proba(X_test_processed)[:, 1]
    test_metrics = compute_classification_metrics(y_test, y_test_pred, y_test_prob, split_name="OOT Test")

    _serialize_artifacts(model, pipeline, project_root)

    return model, pipeline, {"validation": val_metrics, "oot_test": test_metrics}


def _serialize_artifacts(
    model: LogisticRegression,
    pipeline: Pipeline,
    project_root: Path,
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
    project_root : Path
        Root directory of the project for resolving output paths.
    """
    dump_path = project_root / "data" / "processed"
    dump_path.mkdir(parents=True, exist_ok=True)

    pipeline_path = dump_path / "feature_pipeline.joblib"
    model_path = dump_path / "baseline_model.joblib"

    joblib.dump(pipeline, pipeline_path)
    logger.info("Saved feature pipeline → %s", pipeline_path)

    joblib.dump(model, model_path)
    logger.info("Saved baseline model   → %s", model_path)


if __name__ == "__main__":
    train_baseline()