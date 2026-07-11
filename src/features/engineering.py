import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FinancialCleaner(BaseEstimator, TransformerMixin):
    """
    Stateless transformer to strip currency characters ($, ,) and cast strings 
    to float64. Applies exclusively to financial columns.
    """
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless: No statistical parameters to learn.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # 1. Create a defensive copy to prevent SettingWithCopy warnings
        # 2. Iterate through self.columns
        # 3. Apply regex character stripping
        # 4. Cast to float64
        # 5. Return mutated dataframe
        X_copy= X.copy()
        for col in self.columns:
            X_copy[col]= X_copy[col].str.replace(r"[\$,]","",regex=True).astype("float64")
        return X_copy


class NAICSProcessor(BaseEstimator, TransformerMixin):
    """
    Stateless transformer to reduce high-cardinality industry codes by 
    extracting the 2-digit macro-sector and mapping '0' to 'Unknown'.
    """
    def __init__(self, column: str = 'NAICS'):
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless: Explicit mapping rules do not require parameter fitting.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # 1. Create a defensive copy
        # 2. Convert column to string
        # 3. Truncate to first 2 characters
        # 4. Map logical '0' or '00' to explicit 'Unknown' string
        # 5. Return mutated dataframe
        X_copy=X.copy()
        X_copy[self.column]=X_copy[self.column].astype("str").str[0:2].replace(["0","00","na","nan  "],"Unknown")
        return X_copy


class RiskRatioGenerator(BaseEstimator, TransformerMixin):
    """
    Stateless transformer to derive continuous risk ratios (e.g., GovGuaranteeRatio)
    and binary structural flags (e.g., is_real_estate_backed).
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless: Simple vectorized math requires no parameter fitting.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # 1. Create a defensive copy
        # 2. Calculate SBA_Appv / GrAppv
        # 3. Calculate binary flag for Term >= 240
        # 4. Drop the original Term/Approval columns if desired to reduce collinearity
        # 5. Return mutated dataframe
        X_copy= X.copy()
        X_copy["GOV_Ratio"]= np.where(X_copy["GrAppv"]!=0,X_copy["SBA_Appv"]/X_copy["GrAppv"],0.0)
        X_copy["is_backed"]= np.where(X_copy["Term"]>=240,1,0)

        return X_copy


def build_feature_pipeline() -> Pipeline:
    """
    Constructs the master scikit-learn pipeline, chaining custom cleaners 
    with standard statistical transformations.
    """
    # 1. Instantiate custom transformers
    financial_cols = ['DisbursementGross', 'GrAppv', 'SBA_Appv']
    cleaner = FinancialCleaner(columns=financial_cols)
    naics_proc = NAICSProcessor(column='NAICS')
    risk_gen = RiskRatioGenerator()
    
    # 2. Define specific column groupings for statistical scaling
    continuous_features = ['DisbursementGross', 'GrAppv', 'SBA_Appv', 'GOV_Ratio']
    categorical_features = ['NAICS'] # Add 'State' or 'BankState' later if needed
    
    # 3. Instantiate standard ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop' # Drops any columns we haven't explicitly routed (like IDs or raw Dates)
    )
    
    # 4. Chain everything into a final Pipeline object
    master_pipeline = Pipeline(steps=[
        ('financial_cleanup', cleaner),
        ('naics_reduction', naics_proc),
        ('risk_engineering', risk_gen),
        ('statistical_preprocessing', preprocessor)
    ])
    
    return master_pipeline