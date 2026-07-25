import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config.settings import (
    CENTURY_CUTOFF_YEAR,
    POSITIVE_RATE_BOUNDS,
    POST_OUTCOME_COLS,
    RAW_DATA_PATH,
    SPLIT_RATIOS,
)

logger = logging.getLogger(__name__)


class DataIngestor:
    """General ingestion script to perform a chronological split of SBA dataset."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path).resolve()
        self.date_column = "ApprovalDate"
        self.target_column = "MIS_Status"

    def clean(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file missing at {self.file_path}")

        logging.info("Loading and cleaning raw dataset...")
        # low_memory=False prevents pandas from guessing datatypes per chunk
        df = pd.read_csv(self.file_path, low_memory=False)

        # Drop rows missing the target variable
        df = df.dropna(subset=[self.target_column]).copy()

        # Engineer strict binary target: 1 for Default (CHGOFF), 0 for Paid
        df["is_default"] = np.where(df[self.target_column].str.contains("CHGOFF"), 1, 0)

        # A schema or encoding change to MIS_Status (e.g. "CHGOFF" recoded,
        # or the column silently changing dtype) would still produce a
        # binary target here, just a wrong one. Fail loudly instead of
        # training on a corrupted label.
        positive_rate = df["is_default"].mean()
        lower, upper = POSITIVE_RATE_BOUNDS
        if not (lower <= positive_rate <= upper):
            raise ValueError(
                f"Engineered target positive rate {positive_rate:.2%} is outside "
                f"the expected [{lower:.0%}, {upper:.0%}] band — check that "
                f"'{self.target_column}' still encodes charge-offs as 'CHGOFF'."
            )

        # Parse datetime for sorting and offset by 100 years for wrongly converted years
        df[self.date_column] = pd.to_datetime(df[self.date_column], format="mixed")
        mask = df[self.date_column].dt.year > CENTURY_CUTOFF_YEAR
        df.loc[mask, self.date_column] = df.loc[mask, self.date_column] - pd.DateOffset(years=100)

        # Post-outcome columns are only known once a loan's fate is already
        # decided; dropping them here is an explicit invariant, not a
        # side-effect of whichever features the model pipeline happens to
        # select downstream.
        df = df.drop(columns=POST_OUTCOME_COLS, errors="ignore")

        logging.info(f"Loaded {len(df)} records. Base Default Rate: {df['is_default'].mean():.2%}")
        return df

    def chronological_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logging.info("Executing 70/15/15 chronological split...")
        # mergesort is the only stable sort kind in pandas; ~900k rows share
        # relatively few distinct ApprovalDate values, so a non-stable sort
        # would let ties reorder arbitrarily between runs and drift the
        # split boundaries. LoanNr_ChkDgt is the secondary key so the
        # ordering within a date is deterministic too.
        sort_cols = [self.date_column]
        if "LoanNr_ChkDgt" in df.columns:
            sort_cols.append("LoanNr_ChkDgt")
        df_sorted = df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)

        total_rows = len(df_sorted)
        train_frac, val_frac = SPLIT_RATIOS
        train_end = int(total_rows * train_frac)
        val_end = int(total_rows * val_frac)

        train_set = df_sorted.iloc[:train_end].reset_index(drop=True).copy()
        val_set = df_sorted.iloc[train_end:val_end].reset_index(drop=True).copy()
        test_set = df_sorted.iloc[val_end:].reset_index(drop=True).copy()

        logging.info(f"Train: {len(train_set)} | Val: {len(val_set)} | OOT Test: {len(test_set)}")
        return train_set, val_set, test_set


if __name__ == "__main__":
    from src.utils.logging import configure_logging

    configure_logging()

    ingestor = DataIngestor(file_path=RAW_DATA_PATH)
    df_clean = ingestor.clean()
    train_set, val_set, test_set = ingestor.chronological_split(df=df_clean)
