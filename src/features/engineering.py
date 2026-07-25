import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from config.settings import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_VALUE_MAP,
    CONTINUOUS_FEATURES,
    FINANCIAL_COLS,
    PASSTHROUGH_BINARY,
    WINSORIZE_COLS,
)

class FinancialCleaner(BaseEstimator, TransformerMixin):
    """
    Stateless transformer that normalises raw currency strings into float64.

    The SBA National dataset stores disbursement and approval amounts as
    formatted strings (e.g. '$60,000.00'). This transformer strips the
    dollar sign and comma separators before casting to float64, making the
    columns consumable by downstream numeric transformers.

    Inherits from BaseEstimator and TransformerMixin to integrate cleanly
    with scikit-learn Pipeline and cross-validation utilities.

    Parameters
    ----------
    columns : list[str]
        Column names to clean. All listed columns must contain string-typed
        currency values. Non-listed columns are passed through unchanged.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FinancialCleaner":
        """
        Records the full incoming column set (this transformer only
        mutates self.columns in place and passes the rest through
        unchanged, so get_feature_names_out needs to know the whole
        input, not just the columns it touches).
        """
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Strip currency formatting and cast columns to float64.

        A defensive copy is created before mutation to avoid propagating
        changes back to the caller's DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing raw currency string columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with target columns cast to float64.
        """
        X_copy = X.copy()
        for col in self.columns:
            if pd.api.types.is_object_dtype(X_copy[col]) or pd.api.types.is_string_dtype(X_copy[col]):
                X_copy[col] = X_copy[col].str.replace(r"[\$,]", "", regex=True)
            X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
        return X_copy

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(list(input_features))


class NAICSProcessor(BaseEstimator, TransformerMixin):
    """
    Stateless transformer that reduces NAICS industry codes to 2-digit macro-sectors.

    The raw NAICS column contains 6-digit codes (1,312 unique values), which
    would produce an impractically high-cardinality categorical feature.
    Truncating to the first 2 digits collapses these into 20 meaningful
    macro-sector buckets (e.g., '72' = Accommodation & Food Services,
    '44' = Retail Trade), reducing noise and improving model generalisation.

    Rows where NAICS is missing or coded as '0' are mapped to 'Unknown'
    rather than dropped, preserving 201,667 records that would otherwise
    be lost.

    Parameters
    ----------
    column : str, default='NAICS'
        Name of the NAICS column to process.
    """

    def __init__(self, column: str = "NAICS"):
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NAICSProcessor":
        """
        Records the full incoming column set (this transformer only
        rewrites self.column in place and passes the rest through
        unchanged, so get_feature_names_out needs to know the whole
        input, not just the column it touches).
        """
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Truncate 6-digit NAICS codes to 2-digit macro-sector strings.

        Values of '0', '00', and 'na' — which represent missing or
        unclassified industries — are remapped to the sentinel 'Unknown'
        so downstream OneHotEncoder can handle them uniformly via
        handle_unknown='ignore'.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with raw integer NAICS column.

        Returns
        -------
        pd.DataFrame
            DataFrame with NAICS replaced by 2-character sector string.
        """
        
        X[self.column] = (
            X[self.column]
            .astype("str")
            .str[:2]
            .replace({"0": "Unknown", "00": "Unknown", "na": "Unknown"})
        )
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(list(input_features))


class RiskRatioGenerator(BaseEstimator, TransformerMixin):
    """
    Stateless transformer that engineers domain-specific financial risk features.

    Derives two new columns from existing raw fields:

    GOV_Ratio
        The proportion of the total approved loan amount (GrAppv) that is
        guaranteed by the SBA (SBA_Appv). A higher ratio means the lender
        bears less risk, which may correlate with borrower creditworthiness
        or loan type. Division-by-zero is guarded by substituting 0.0 when
        GrAppv is zero.

    is_backed
        Binary flag indicating whether the loan term is >= 240 months
        (20 years). Long-term loans in the SBA dataset are typically
        real-estate backed (SBA 504 programme), which carry a structurally
        different risk profile from short-term working capital loans.
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RiskRatioGenerator":
        """
        No-op. All derived features are computed purely from the input
        row values with no training-set statistics required.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute GOV_Ratio and is_backed and append them to the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame. Must contain 'GrAppv', 'SBA_Appv' (as float64,
            post-FinancialCleaner) and 'Term' (integer months).

        Returns
        -------
        pd.DataFrame
            DataFrame with two additional columns: 'GOV_Ratio' (float64)
            and 'is_backed' (int, 0 or 1).
        """
        
        X["GOV_Ratio"] = np.where(
            X["GrAppv"] != 0,
            X["SBA_Appv"] / X["GrAppv"],
            0.0,
        )
        X["is_backed"] = np.where(X["Term"] >= 240, 1, 0)
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = []
        return np.asarray(list(input_features) + ["GOV_Ratio", "is_backed"])

class CategoricalSanitizer(BaseEstimator, TransformerMixin):
    """
    Stateless transformer that normalises dirty categorical columns into
    a consistent vocabulary before one-hot encoding.

    The SBA National dataset contains known encoding inconsistencies in
    flag columns: mixed case ('y'/'Y'), numeric placeholders ('0'), and
    undocumented sentinel values ('T', 'S', 'C'). Passing these raw to
    OneHotEncoder would create spurious extra categories and inflate the
    feature matrix.

    This transformer applies an explicit allow-list mapping per column.
    Any value not present in the mapping dict is filled with the 'Unknown'
    sentinel, which OneHotEncoder then handles uniformly via
    handle_unknown='ignore'.

    Parameters
    ----------
    column_map : dict[str, dict[str, str]]
        Mapping of column_name -> {raw_value: canonical_value}.
        Every key in the outer dict must be a column present in the
        DataFrame passed to transform().
    """

    def __init__(self, column_map: dict[str, dict[str, str]]):
        self.column_map = column_map

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalSanitizer":
        """
        Records the full incoming column set (this transformer only
        remaps the columns in column_map in place and passes the rest
        through unchanged, so get_feature_names_out needs to know the
        whole input, not just the columns it touches).
        """
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply allow-list mappings to each registered column.

        Columns in X that are not present in column_map are passed through
        unchanged, making this transformer safe to compose with datasets
        that have varying column sets.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing raw categorical columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with each mapped column replaced by its canonical
            string category. Unmapped values become 'Unknown'.
        """
        
        for col, mapping_dict in self.column_map.items():
            if col in X.columns:
                X[col] = (
                    X[col]
                    .astype(str)
                    .map(mapping_dict)
                    .fillna("Unknown")
                )
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(list(input_features))


class FranchiseEncoder(BaseEstimator, TransformerMixin):
    """
    Stateless transformer that binarises the high-cardinality FranchiseCode
    column into a single interpretable flag.

    FranchiseCode contains thousands of distinct franchise identifiers,
    making it unsuitable for direct one-hot encoding. The meaningful
    business distinction is binary: the borrower is either franchised
    (code > 1) or not (code 0 or 1, where 1 is the SBA's placeholder for
    non-franchise). The original FranchiseCode column is retained in the
    DataFrame but falls through the ColumnTransformer's remainder='drop'.

    Parameters
    ----------
    column : str, default='FranchiseCode'
        Name of the franchise code column.
    threshold : int, default=1
        Codes strictly greater than this value are treated as franchised.
        Codes 0 and 1 both map to non-franchised (0).
    """

    def __init__(self, column: str = "FranchiseCode", threshold: int = 1):
        self.column = column
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FranchiseEncoder":
        """
        No-op. The binarisation threshold is fixed at construction time.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Append the binary 'is_franchise' column derived from FranchiseCode.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing the FranchiseCode column.

        Returns
        -------
        pd.DataFrame
            DataFrame with an additional 'is_franchise' column (int, 0 or 1).
        """
        
        X["is_franchise"] = np.where(X[self.column] > self.threshold, 1, 0)
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = []
        return np.asarray(list(input_features) + ["is_franchise"])


class OutlierWinsorizer(BaseEstimator, TransformerMixin):
    """
    Stateful transformer that clips continuous features to percentile bounds
    learned from the training set.

    SBA loan amounts span several orders of magnitude — a single $10M
    loan sits alongside a median of ~$60K. These extremes distort
    StandardScaler's mean and variance estimates, compressing the effective
    signal range for the vast majority of observations. Winsorizing clips
    values outside the [lower_quantile, upper_quantile] range to the
    boundary values rather than removing the records.

    Critically, bounds are computed exclusively on the training fold and
    applied as a fixed transform to validation and OOT sets, preventing
    data leakage. GOV_Ratio is intentionally excluded from winsorization
    because it is a ratio bounded [0.0, 1.0] by construction.

    Parameters
    ----------
    columns : list[str]
        Continuous columns to winsorize. Should match the columns passed
        to StandardScaler in the downstream ColumnTransformer.
    lower_quantile : float, default=0.01
        Lower clipping percentile (values below this are raised to the
        boundary).
    upper_quantile : float, default=0.99
        Upper clipping percentile (values above this are lowered to the
        boundary).

    Attributes
    ----------
    bounds_ : dict[str, tuple[float, float]]
        Per-column (lower, upper) clip bounds fitted on the training data.
        Only available after fit() has been called.
    """

    def __init__(
        self,
        columns: list[str],
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ):
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OutlierWinsorizer":
        """
        Compute and store per-column clip bounds from the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training DataFrame. Only self.columns are read; all others
            are ignored.
        y : pd.Series or None
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        OutlierWinsorizer
            Fitted transformer with bounds_ populated.
        """
        self.bounds_: dict[str, tuple[float, float]] = {
            col: (
                X[col].quantile(self.lower_quantile),
                X[col].quantile(self.upper_quantile),
            )
            for col in self.columns
        }
        # Full incoming column set: this transformer only clips self.columns
        # in place and passes the rest through unchanged, so
        # get_feature_names_out needs to know the whole input, not just
        # the columns it touches.
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clip each column to its training-set percentile bounds.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame (train, validation, or OOT). Must contain all
            columns listed in self.columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with extreme values clipped to [lower, upper] per column.

        Raises 
        ------
        NotFittedError if transform is called before fit()
        """
        check_is_fitted(self, "bounds_")
        
        for col in self.columns:
            lower, upper = self.bounds_[col]
            X[col] = np.clip(X[col], lower, upper)
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(list(input_features))


def _build_domain_stages() -> list[tuple[str, TransformerMixin]]:
    """
    Build the six domain-specific transformer stages shared by both the
    logistic-regression and tree-model feature pipelines.

    build_feature_pipeline() and build_tree_feature_pipeline() differ only
    in their final ColumnTransformer (scaled + one-hot vs. passthrough +
    ordinal), so this prefix is defined once and reused by both.

    Returns
    -------
    list[tuple[str, TransformerMixin]]
        Named steps ready to prepend to a Pipeline, followed by whichever
        final ColumnTransformer the caller assembles.
    """
    cleaner = FinancialCleaner(columns=FINANCIAL_COLS)
    naics_proc = NAICSProcessor(column="NAICS")
    risk_gen = RiskRatioGenerator()

    # RevLineCr and LowDoc: canonical values are Y/N; dirty values include
    # lowercase variants and undocumented sentinels ('T', 'S', 'C', '0').
    # NewExist: stored as float64 in the CSV, so str() yields '1.0'/'2.0'.
    cat_sanitizer = CategoricalSanitizer(column_map=CATEGORICAL_VALUE_MAP)

    # FranchiseCode > 1 indicates an active franchise relationship.
    # Codes 0 and 1 are the SBA's placeholders for non-franchise loans.
    franchise_enc = FranchiseEncoder(column="FranchiseCode", threshold=1)

    # GOV_Ratio is excluded: bounded [0, 1] by construction in RiskRatioGenerator.
    # is_franchise is excluded: binary flag, winsorizing would be a no-op.
    winsorizer = OutlierWinsorizer(
        columns=WINSORIZE_COLS,
        lower_quantile=0.01,
        upper_quantile=0.99,
    )

    return [
        ("financial_cleanup", cleaner),
        ("naics_reduction", naics_proc),
        ("risk_engineering", risk_gen),
        ("categorical_sanitization", cat_sanitizer),
        ("franchise_encoding", franchise_enc),
        ("outlier_winsorization", winsorizer),
    ]


def build_feature_pipeline() -> Pipeline:
    """
    Assemble the full preprocessing pipeline for the SBA credit scoring model.

    Chains the shared six-stage domain prefix (see _build_domain_stages())
    with a final ColumnTransformer that scales continuous features and
    one-hot encodes categoricals — the encoding a linear model needs. The
    pipeline is designed to be fit exclusively on the training fold and
    applied as a stateless transform to validation and OOT sets,
    preventing data leakage at every stage.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn Pipeline. Call .fit_transform(X_train) once,
        then .transform() on validation and OOT sets.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("pass", StandardScaler(), PASSTHROUGH_BINARY),
        ],
        remainder="drop",  # Discards IDs, dates, raw target columns, and FranchiseCode
    )

    master_pipeline = Pipeline(steps=[
        *_build_domain_stages(),
        ("statistical_preprocessing", preprocessor),
    ])

    return master_pipeline


def build_tree_feature_pipeline() -> Pipeline:
    """
    Assemble a tree-model-optimised preprocessing pipeline.

    Reuses the same six domain-specific transformer stages as
    build_feature_pipeline() (see _build_domain_stages()), but replaces
    the final ColumnTransformer with a tree-friendly variant:

        - No StandardScaler (trees are scale-invariant)
        - OrdinalEncoder instead of OneHotEncoder (preserves split
          efficiency and reduces dimensionality)
        - passthrough for continuous and binary features

    This pipeline is designed for XGBoost and LightGBM consumers.
    The existing build_feature_pipeline() remains the Phase 1 artifact
    for the Logistic Regression baseline and must not be modified.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn Pipeline. Call .fit_transform(X_train) once,
        then .transform() on validation and OOT sets.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", CONTINUOUS_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
            ("pass", "passthrough", PASSTHROUGH_BINARY),
        ],
        remainder="drop",
    )

    tree_pipeline = Pipeline(steps=[
        *_build_domain_stages(),
        ("tree_preprocessing", preprocessor),
    ])

    return tree_pipeline
