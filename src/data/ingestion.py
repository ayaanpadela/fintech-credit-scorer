import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

# Configure basic logging for telemetry
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

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
        
        # Parse datetime for sorting
        df[self.date_column] = pd.to_datetime(df[self.date_column], format='mixed')
        
        logging.info(f"Loaded {len(df)} records. Base Default Rate: {df['is_default'].mean():.2%}")
        return df
    
    def chronological_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logging.info("Executing 70/15/15 chronological split...")
        df_sorted = df.sort_values(by=self.date_column).reset_index(drop=True)
        
        total_rows = len(df_sorted)
        train_end = int(total_rows * 0.7)
        val_end = int(total_rows * 0.85)
        
        train_set = df_sorted.iloc[:train_end].copy()
        val_set = df_sorted.iloc[train_end:val_end].copy()
        test_set = df_sorted.iloc[val_end:].copy()
        
        logging.info(f"Train: {len(train_set)} | Val: {len(val_set)} | OOT Test: {len(test_set)}")
        return train_set, val_set, test_set

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "SBAnational.csv"
    
    ingestor = DataIngestor(file_path=data_path)
    df_clean = ingestor.clean()
    train_set, val_set, test_set = ingestor.chronological_split(df=df_clean)