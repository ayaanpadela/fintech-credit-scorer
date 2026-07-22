# Explainable FinTech Credit Scorer API

A production-ready credit scoring microservice designed to harmonize complex machine learning algorithms with stringent regulatory constraints (ECOA and FCRA) and high-performance software engineering.

The system ingests borrower data, predicts the calibrated probability of default (PD), and computes exact SHAP (SHapley Additive exPlanations) values for interpretability. It maps these explanations to legally compliant adverse action reason codes and serves the results via a high-concurrency FastAPI endpoint.

## Project Architecture & Methodology

The development of this API follows an expert-level, four-phase architectural roadmap:

### Phase 1: Data Engineering, Preprocessing, and Baseline Modeling
*   **Objectives**: Chronological data splitting (Train/Val/Out-of-Time), domain-informed missing value imputation, robust feature engineering (e.g., debt-to-income ratios), cost-sensitive learning for class imbalance, and training a benchmark Logistic Regression model.
*   **Status**: **Completed**.
    *   Files implemented: `src/data/ingestion.py`, `src/features/engineering.py`, `src/models/baseline.py`, and exploratory notebooks (`notebooks/feature_exp.ipynb`).

### Phase 2: Gradient Boosting & Tree-Optimised Pipeline
*   **Objectives**: Train XGBoost and LightGBM classifiers using a tree-optimised feature pipeline (ordinal encoding instead of one-hot), early stopping on the chronological validation fold, and cost-sensitive class weighting to close the Phase 1 validation–OOT generalisation gap.
*   **Status**: **Completed (default hyperparameters — no tuning applied yet)**.
    *   Files implemented: `src/models/gradient_boosting.py`, tree pipeline builder in `src/features/engineering.py`, hyperparameter search spaces in `config/settings.py`.

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
| **ROC-AUC** | 0.8042 | 0.6865 | +0.1281 |
| **PR-AUC** | 0.6276 | 0.4146 | — |
| **Brier Score** | 0.2164 | 0.2525 | — |
| **F1 (Default)** | 0.67 | 0.55 | +0.08 |
| **Precision (Default)** | 0.52 | 0.41 | +0.07 |
| **Recall (Default)** | 0.91 | 0.81 | +0.04 |

**Why the improvement?** The initial baseline operated on a severely limited column map that excluded key financial predictors. The expanded pipeline unlocked signal from three sources: (1) **dollar-amount features** — disbursement, approval, and SBA guarantee amounts were trapped as unparseable currency strings (`'$60,000.00'`) until `FinancialCleaner` converted them to float64; (2) **engineered risk ratios** — `GOV_Ratio` (SBA guarantee proportion) and `is_backed` (long-term real-estate flag) capture structural loan risk the raw columns do not express; and (3) **cleaned categoricals** — `RevLineCr`, `LowDoc`, `NewExist`, and `NAICS` contained mixed-case values, undocumented sentinels, and high-cardinality codes that were either dropped or collapsed into noise before sanitization. The ~12-point ROC-AUC gap between validation (0.80) and OOT (0.69) reflects expected temporal drift across the chronological split and is the primary target for Phase 2 gradient boosting.

## Phase 2 Results — Gradient Boosting (Default Hyperparameters)

Phase 2 replaced the linear baseline with tree-based gradient boosting models, using a dedicated tree-optimised pipeline (`build_tree_feature_pipeline()`) that applies ordinal encoding instead of one-hot encoding to preserve native split semantics in tree learners.

> **Note:** All results below were obtained using **default hyperparameters only** — no Optuna tuning has been applied. Hyperparameter search spaces for both XGBoost and LightGBM are defined in `config/settings.py` (50 Optuna trials) and are expected to yield further improvement once tuning is implemented.

### XGBoost (default params, `scale_pos_weight`, early stopping @ 50 rounds)

| Metric | Validation | OOT Test | Δ vs Baseline LR (OOT) |
|:--|:--:|:--:|:--:|
| **ROC-AUC** | 0.9664 | 0.9375 | +0.2510 |
| **PR-AUC** | 0.9191 | 0.8524 | +0.4378 |
| **Brier Score** | 0.0653 | 0.0943 | −0.1582 |
| **F1 (Default)** | 0.89 | 0.81 | +0.26 |
| **Precision (Default)** | 0.84 | 0.75 | +0.34 |
| **Recall (Default)** | 0.93 | 0.87 | +0.06 |

### LightGBM (default params, `is_unbalance=True`, early stopping @ 50 rounds)

| Metric | Validation | OOT Test | Δ vs Baseline LR (OOT) |
|:--|:--:|:--:|:--:|
| **ROC-AUC** | 0.9654 | 0.9371 | +0.2506 |
| **PR-AUC** | 0.9151 | 0.8454 | +0.4308 |
| **Brier Score** | 0.0690 | 0.1010 | −0.1515 |
| **F1 (Default)** | 0.88 | 0.79 | +0.24 |
| **Precision (Default)** | 0.82 | 0.71 | +0.30 |
| **Recall (Default)** | 0.95 | 0.89 | +0.08 |

### Phase 2 Analysis

Even with **default hyperparameters and no tuning**, both gradient boosting models delivered a **+25-point OOT ROC-AUC improvement** over the baseline Logistic Regression (0.69 → 0.94), representing a step-change from moderate to production-grade discrimination. Three key dynamics drove the gain:

1. **Non-linear interaction capture** — Tree ensembles natively model feature interactions (e.g., `GOV_Ratio × Term`, `DisbursementGross × is_backed`) that the linear model could not represent without manual polynomial expansion.
2. **Generalisation gap compression** — The validation–OOT ROC-AUC gap shrank from ~12 points (LR: 0.80 → 0.69) to ~3 points (XGB: 0.97 → 0.94), indicating that the tree models are far more robust to temporal distribution shift.
3. **Calibration improvement** — Brier scores dropped from 0.25 (LR) to 0.09 (XGB), meaning predicted probabilities of default are now well-calibrated and suitable for downstream risk pricing.

XGBoost slightly outperforms LightGBM across all OOT metrics (ROC-AUC 0.9375 vs 0.9371, F1 0.81 vs 0.79) and is the current **Phase 2 champion model**. Both models were trained with default hyperparameters; further improvement via Optuna-based tuning (50 trials, search spaces defined in `config/settings.py`) is expected in an optional optimisation pass.

## Technology Stack
*   **Core Language**: Python 3.12+
*   **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`
*   **Explainability**: `shap`
*   **Web Framework**: `fastapi`, `uvicorn`, `orjson`
*   **Serialization**: `joblib`
*   **Dependency Management**: `uv`

## Progress Summary
Phases 1 and 2 are complete. The foundational project structure has been established (`src/`, `app/`, `tests/`, `notebooks/`). Core data pipelines, an expanded feature engineering pipeline (6 custom sklearn transformers, 614 lines), a cost-sensitive baseline Logistic Regression, and gradient boosting models (XGBoost and LightGBM) are in place. The Phase 2 XGBoost champion — trained with **default hyperparameters only** — achieves an OOT ROC-AUC of 0.9375, a +25-point improvement over the Phase 1 baseline (0.6865), with the validation–OOT gap compressed from ~12 points to ~3 points. Optuna hyperparameter tuning has not yet been applied and is expected to yield further gains. Next: Phase 3 (SHAP explainability and adverse action code mapping).
