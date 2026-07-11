# Explainable FinTech Credit Scorer API

A production-ready credit scoring microservice designed to harmonize complex machine learning algorithms with stringent regulatory constraints (ECOA and FCRA) and high-performance software engineering.

The system ingests borrower data, predicts the calibrated probability of default (PD), and computes exact SHAP (SHapley Additive exPlanations) values for interpretability. It maps these explanations to legally compliant adverse action reason codes and serves the results via a high-concurrency FastAPI endpoint.

## Project Architecture & Methodology

The development of this API follows an expert-level, four-phase architectural roadmap:

### Phase 1: Data Engineering, Preprocessing, and Baseline Modeling
*   **Objectives**: Chronological data splitting (Train/Val/Out-of-Time), domain-informed missing value imputation, robust feature engineering (e.g., debt-to-income ratios), cost-sensitive learning for class imbalance, and training a benchmark Logistic Regression model.
*   **Status**: **In Progress / Completed**. 
    *   Files implemented: `src/data/ingestion.py`, `src/features/engineering.py`, `src/models/baseline.py`, and exploratory notebooks (`notebooks/feature_exp.ipynb`).

## Baseline Results
The baseline Logistic Regression model has been trained and evaluated using a chronologically split Out-of-Time (OOT) test set.

**Dataset Split:**
*   Total Records: 897,167 (Base Default Rate: 17.56%)
*   Train: 628,016
*   Validation: 134,575
*   OOT Test: 134,576

**Validation Set Performance:**
*   **ROC-AUC**: 0.5643
*   **F1-Score (Default Class)**: 0.53 (Precision: 0.39, Recall: 0.85)

**Out-of-Time (OOT) Test Set Performance:**
*   **ROC-AUC**: 0.5584
*   **F1-Score (Default Class)**: 0.47 (Precision: 0.34, Recall: 0.77)

## Technology Stack
*   **Core Language**: Python 3.10+
*   **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`
*   **Explainability**: `shap`
*   **Web Framework**: `fastapi`, `uvicorn`, `orjson`
*   **Serialization**: `joblib`
*   **Dependency Management**: `poetry` / `uv`

## Progress Summary
The foundational project structure has been established (`src/`, `app/`, `tests/`, `notebooks/`). Core data pipelines, feature engineering transformations, and baseline modeling scripts are in place, successfully setting up the robust chronological validation framework required for subsequent gradient boosting and API development.
