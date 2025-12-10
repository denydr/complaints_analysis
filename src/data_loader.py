"""
data_loader.py
--------------

Functions for loading complaints data from disk into pandas DataFrames.

Currently supports:
- Loading the raw complaints CSV from data/raw/.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from config import RAW_COMPLAINTS_FILE


def load_raw_complaints(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw complaints dataset.

    Parameters
    ----------
    path : Optional[Path]
        Custom path to a CSV file.
        If None, uses the default RAW_COMPLAINTS_FILE from config.py.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw complaints.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    csv_path = path or RAW_COMPLAINTS_FILE

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw complaints file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Expect at least a description column, optionally an ID
    required_cols = ["description"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in complaints CSV: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    # Optionally, ensure descriptions are strings
    df["description"] = df["description"].astype(str)

    return df


if __name__ == "__main__":
    # Quick manual test: run `python -m src.data_loader` from project root
    df_test = load_raw_complaints()
    print(df_test.head())
    print(f"\nLoaded {len(df_test)} rows from raw complaints file.")