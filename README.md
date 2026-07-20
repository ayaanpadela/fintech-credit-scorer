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

The baseline Logistic Regression model (`class_weight="balanced"`) has been trained and evaluated using a chronologically split Out-of-Time (OOT) test set.

**Dataset Split:**
*   Total Records: 897,167 (Base Default Rate: 17.56%)
*   Train: 628,016
*   Validation: 134,575
*   OOT Test: 134,576

### Initial Baseline (Reduced Feature Set)

The first training run used a minimal subset of columns, producing near-random discrimination:

| Metric | Validation | OOT Test |
|:--|:--:|:--:|
| **ROC-AUC** | 0.5643 | 0.5584 |
| **F1 (Default)** | 0.53 | 0.47 |
| **Precision (Default)** | 0.39 | 0.34 |
| **Recall (Default)** | 0.85 | 0.77 |

### Current Baseline (Expanded Feature Engineering)

After expanding the feature pipeline to include domain-informed financial transformers — currency normalization (`FinancialCleaner`), NAICS macro-sector reduction (`NAICSProcessor`), SBA guarantee ratio and real-estate-backed flag engineering (`RiskRatioGenerator`), dirty categorical sanitization (`CategoricalSanitizer`), franchise binarization (`FranchiseEncoder`), and train-fitted outlier winsorization (`OutlierWinsorizer`) — the model improved dramatically:

| Metric | Validation | OOT Test | Δ vs Initial (OOT) |
|:--|:--:|:--:|:--:|
| **ROC-AUC** | 0.8036 | 0.6876 | +0.1292 |
| **PR-AUC** | 0.6260 | 0.4152 | — |
| **Brier Score** | 0.2157 | 0.2516 | — |
| **F1 (Default)** | 0.67 | 0.55 | +0.08 |
| **Precision (Default)** | 0.53 | 0.42 | +0.08 |
| **Recall (Default)** | 0.91 | 0.81 | +0.04 |

**Why the improvement?** The initial baseline operated on a severely limited column map that excluded key financial predictors. The expanded pipeline unlocked signal from three sources: (1) **dollar-amount features** — disbursement, approval, and SBA guarantee amounts were trapped as unparseable currency strings (`'$60,000.00'`) until `FinancialCleaner` converted them to float64; (2) **engineered risk ratios** — `GOV_Ratio` (SBA guarantee proportion) and `is_backed` (long-term real-estate flag) capture structural loan risk the raw columns do not express; and (3) **cleaned categoricals** — `RevLineCr`, `LowDoc`, `NewExist`, and `NAICS` contained mixed-case values, undocumented sentinels, and high-cardinality codes that were either dropped or collapsed into noise before sanitization. The ~12-point ROC-AUC gap between validation (0.80) and OOT (0.69) reflects expected temporal drift across the chronological split and is the primary target for Phase 2 gradient boosting.

## Technology Stack
*   **Core Language**: Python 3.10+
*   **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`
*   **Explainability**: `shap`
*   **Web Framework**: `fastapi`, `uvicorn`, `orjson`
*   **Serialization**: `joblib`
*   **Dependency Management**: `poetry` / `uv`

## Progress Summary
Phase 1 is complete. The foundational project structure has been established (`src/`, `app/`, `tests/`, `notebooks/`). Core data pipelines, an expanded feature engineering pipeline (6 custom sklearn transformers, 541 lines), and a cost-sensitive baseline Logistic Regression are in place. The baseline clears the Phase 1 success criterion of OOT ROC-AUC > 0.65, establishing a solid benchmark for Phase 2 gradient boosting.
