import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
        No-op. This transformer is stateless — all cleaning rules are
        deterministic and require no parameters learned from the data.
        """
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
            X_copy[col] = X_copy[col].str.replace(r"[\$,]", "", regex=True).astype("float64")
        return X_copy


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
        No-op. Mapping rules are deterministic and require no training-set
        statistics.
        """
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
        X_copy = X.copy()
        X_copy[self.column] = (
            X_copy[self.column]
            .astype("str")
            .str[:2]
            .replace({"0": "Unknown", "00": "Unknown", "na": "Unknown"})
        )
        return X_copy


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
        X_copy = X.copy()
        X_copy["GOV_Ratio"] = np.where(
            X_copy["GrAppv"] != 0,
            X_copy["SBA_Appv"] / X_copy["GrAppv"],
            0.0,
        )
        X_copy["is_backed"] = np.where(X_copy["Term"] >= 240, 1, 0)
        return X_copy


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
        No-op. The allow-list mapping is fully specified at construction
        time and requires no statistics from the training data.
        """
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
        X_copy = X.copy()
        for col, mapping_dict in self.column_map.items():
            if col in X_copy.columns:
                X_copy[col] = (
                    X_copy[col]
                    .astype(str)
                    .map(mapping_dict)
                    .fillna("Unknown")
                )
        return X_copy


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
        X_copy = X.copy()
        X_copy["is_franchise"] = np.where(X_copy[self.column] > self.threshold, 1, 0)
        return X_copy


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
        KeyError
            If transform() is called before fit(), or if X is missing a
            column that was present during fit().
        """
        X_copy = X.copy()
        for col in self.columns:
            lower, upper = self.bounds_[col]
            X_copy[col] = np.clip(X_copy[col], lower, upper)
        return X_copy


def build_feature_pipeline() -> Pipeline:
    """
    Assemble the full preprocessing pipeline for the SBA credit scoring model.

    Chains seven sequential stages into a single scikit-learn Pipeline
    object. The pipeline is designed to be fit exclusively on the training
    fold and applied as a stateless transform to validation and OOT sets,
    preventing data leakage at every stage.

    Pipeline stages (in execution order)
    --------------------------------------
    1. financial_cleanup
        FinancialCleaner — strips '$' and ',' from DisbursementGross,
        GrAppv, SBA_Appv and casts them to float64.

    2. naics_reduction
        NAICSProcessor — truncates 6-digit NAICS codes to 2-digit
        macro-sector strings; maps missing/zero codes to 'Unknown'.

    3. risk_engineering
        RiskRatioGenerator — derives GOV_Ratio (SBA guarantee proportion)
        and is_backed (long-term / real-estate-backed loan flag).

    4. categorical_sanitization
        CategoricalSanitizer — normalises RevLineCr, LowDoc, and NewExist
        to a clean vocabulary, collapsing mixed-case and dirty sentinel
        values to 'Unknown'.

    5. franchise_encoding
        FranchiseEncoder — collapses FranchiseCode to a binary is_franchise
        flag. The original FranchiseCode column is dropped by the
        ColumnTransformer's remainder='drop'.

    6. outlier_winsorization
        OutlierWinsorizer — clips the seven raw continuous columns to
        [1st, 99th] percentile bounds learned from the training set.
        GOV_Ratio is excluded because it is bounded [0, 1] by construction.

    7. statistical_preprocessing
        ColumnTransformer routing:
            - StandardScaler  → 8 continuous features
            - OneHotEncoder   → 4 categorical features (NAICS, RevLineCr,
                                LowDoc, NewExist)
            - passthrough      → 3 binary flags (is_backed, UrbanRural,
                                is_franchise)
            - remainder='drop' → all other columns (IDs, dates, raw targets)

    Returns
    -------
    Pipeline
        Unfitted scikit-learn Pipeline. Call .fit_transform(X_train) once,
        then .transform() on validation and OOT sets.
    """
    # --- Transformer instantiation ---

    financial_cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
    cleaner = FinancialCleaner(columns=financial_cols)
    naics_proc = NAICSProcessor(column="NAICS")
    risk_gen = RiskRatioGenerator()

    # RevLineCr and LowDoc: canonical values are Y/N; dirty values include
    # lowercase variants and undocumented sentinels ('T', 'S', 'C', '0').
    # NewExist: stored as float64 in the CSV, so str() yields '1.0'/'2.0'.
    cat_sanitizer = CategoricalSanitizer(column_map={
        "RevLineCr": {
            "Y": "Y", "N": "N",
            "y": "Y", "n": "N",
            "0": "Unknown", "T": "Unknown",
            "nan": "Unknown",
        },
        "LowDoc": {
            "Y": "Y", "N": "N",
            "y": "Y", "n": "N",
            "S": "Unknown", "C": "Unknown",
            "0": "Unknown",
            "nan": "Unknown",
        },
        "NewExist": {
            "1": "Existing", "1.0": "Existing",
            "2": "New",      "2.0": "New",
            "0": "Unknown",
            "nan": "Unknown",
        },
    })

    # FranchiseCode > 1 indicates an active franchise relationship.
    # Codes 0 and 1 are the SBA's placeholders for non-franchise loans.
    franchise_enc = FranchiseEncoder(column="FranchiseCode", threshold=1)

    # GOV_Ratio is excluded: bounded [0, 1] by construction in RiskRatioGenerator.
    # is_franchise is excluded: binary flag, winsorizing would be a no-op.
    winsorizer = OutlierWinsorizer(
        columns=["DisbursementGross", "GrAppv", "SBA_Appv", "Term", "NoEmp", "CreateJob", "RetainedJob"],
        lower_quantile=0.01,
        upper_quantile=0.99,
    )

    # --- ColumnTransformer column routing ---

    # Scaled continuous features — post-cleaning and post-winsorization.
    continuous_features = [
        "DisbursementGross",  # Total gross disbursement amount
        "GrAppv",             # Gross amount approved by bank
        "SBA_Appv",           # Amount guaranteed by SBA
        "GOV_Ratio",          # SBA guarantee proportion (engineered)
        "Term",               # Loan term in months
        "NoEmp",              # Number of employees at origination
        "CreateJob",          # Projected jobs to be created
        "RetainedJob",        # Projected jobs to be retained
    ]

    # One-hot encoded categoricals — post-sanitization.
    # handle_unknown='ignore' silently drops unseen categories at inference,
    # which is the correct behaviour for OOT data.
    categorical_features = [
        "NAICS",      # 2-digit macro-sector (post NAICSProcessor)
        "RevLineCr",  # Revolving line of credit flag (post CategoricalSanitizer)
        "LowDoc",     # LowDoc programme flag (post CategoricalSanitizer)
        "NewExist",   # New vs existing business (post CategoricalSanitizer)
    ]

    # Binary flags passed through without transformation.
    # UrbanRural (0/1/2) is ordinal by SBA definition and treated as
    # a raw numeric pass-through rather than one-hot encoded.
    passthrough_binary = [
        "is_backed",    # Long-term / real-estate backed loan (from RiskRatioGenerator)
        "UrbanRural",   # Urban (1) / Rural (2) / Undefined (0)
        "is_franchise", # Franchised borrower flag (from FranchiseEncoder)
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  StandardScaler(),                                          continuous_features),
            ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("pass", "passthrough",                                             passthrough_binary),
        ],
        remainder="drop",  # Discards IDs, dates, raw target columns, and FranchiseCode
    )

    # --- Final pipeline assembly ---

    master_pipeline = Pipeline(steps=[
        ("financial_cleanup",        cleaner),
        ("naics_reduction",          naics_proc),
        ("risk_engineering",         risk_gen),
        ("categorical_sanitization", cat_sanitizer),
        ("franchise_encoding",       franchise_enc),
        ("outlier_winsorization",    winsorizer),
        ("statistical_preprocessing", preprocessor),
    ])

    return master_pipeline