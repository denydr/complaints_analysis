"""
data_loader.py
--------------

Functions for loading complaints data from disk into pandas DataFrames.

Currently supports:
- Loading the raw complaints CSV from data/raw/.
- Loading the cleaned complaints CSV from data/cleaned/ for downstream steps
  (vectorization and topic modeling).

Notes
-----
- LDA vectorization must use ONLY the 'lda_description' column from the cleaned file.
- BERTopic uses 'bertopic_description' directly (no vectorization required).
- Topic modeling later may load either:
  - cleaned text (for BERTopic), or
  - vectorized matrices + vocab (for LDA).
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import load_npz
from src.config import (
    RAW_COMPLAINTS_FILE,
    CLEANED_COMPLAINTS_FILE,
    LDA_BOW_MATRIX_FILE,
    LDA_TFIDF_MATRIX_FILE,
    LDA_BOW_FEATURE_NAMES_FILE,
    LDA_TFIDF_FEATURE_NAMES_FILE,
    LDA_DOC_IDS_FILE,
    LDA_PROCESSED_TEXT_FILE,
)

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
    df["description"] = df["description"].fillna("").astype(str)

    return df


def load_cleaned_complaints(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the cleaned complaints dataset produced by cleaning.py.

    Parameters
    ----------
    path : Optional[Path]
        Custom path to a cleaned CSV file.
        If None, uses CLEANED_COMPLAINTS_FILE from config.py.

    Returns
    -------
    pd.DataFrame
        DataFrame containing cleaned complaints columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    csv_path = path or CLEANED_COMPLAINTS_FILE

    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned complaints file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Required cleaned columns for downstream processing
    required_cols = ["description", "lda_description", "bertopic_description"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in cleaned complaints CSV: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    # Ensure string types
    df["description"] = df["description"].fillna("").astype(str)
    df["lda_description"] = df["lda_description"].fillna("").astype(str)
    df["bertopic_description"] = df["bertopic_description"].fillna("").astype(str)

    return df


def load_cleaned_for_lda(
    path: Optional[Path] = None,
    id_column: str = "service_request_id",
    keep_id: bool = True,
) -> pd.DataFrame:
    """
    Convenience loader for LDA vectorization.

    Returns a DataFrame containing only the columns needed for LDA vectorization:
    - service_request_id (optional, if present and keep_id=True)
    - lda_description (required)

    Parameters
    ----------
    path : Optional[Path]
        Custom path to a cleaned CSV file.
        If None, uses CLEANED_COMPLAINTS_FILE from config.py.
    id_column : str
        Name of the ID column to keep if it exists.
    keep_id : bool
        If True and the ID column exists, include it in the output.

    Returns
    -------
    pd.DataFrame
        DataFrame ready for LDA vectorization.

    Raises
    ------
    ValueError
        If 'lda_description' is missing.
    """
    df = load_cleaned_complaints(path=path)

    if "lda_description" not in df.columns:
        raise ValueError("Expected column 'lda_description' in cleaned complaints file.")

    cols_to_keep = ["lda_description"]
    if keep_id and id_column in df.columns:
        cols_to_keep.insert(0, id_column)

    df_out = df[cols_to_keep].copy()
    df_out["lda_description"] = df_out["lda_description"].fillna("").astype(str)
    df_out = df_out[df_out["lda_description"].str.strip().ne("")].reset_index(drop=True)

    return df_out


def load_cleaned_for_bertopic(
    path: Optional[Path] = None,
    id_column: str = "service_request_id",
    keep_id: bool = True,
) -> pd.DataFrame:
    """
    Convenience loader for BERTopic.

    Returns a DataFrame containing only the columns needed for BERTopic:
    - service_request_id (optional, if present and keep_id=True)
    - bertopic_description (required)

    Parameters
    ----------
    path : Optional[Path]
        Custom path to a cleaned CSV file.
        If None, uses CLEANED_COMPLAINTS_FILE from config.py.
    id_column : str
        Name of the ID column to keep if it exists.
    keep_id : bool
        If True and the ID column exists, include it in the output.

    Returns
    -------
    pd.DataFrame
        DataFrame ready for BERTopic.

    Raises
    ------
    ValueError
        If 'bertopic_description' is missing.
    """
    df = load_cleaned_complaints(path=path)

    if "bertopic_description" not in df.columns:
        raise ValueError("Expected column 'bertopic_description' in cleaned complaints file.")

    cols_to_keep = ["bertopic_description"]
    if keep_id and id_column in df.columns:
        cols_to_keep.insert(0, id_column)

    df_out = df[cols_to_keep].copy()
    df_out["bertopic_description"] = df_out["bertopic_description"].fillna("").astype(str)
    df_out = df_out[df_out["bertopic_description"].str.strip().ne("")].reset_index(drop=True)

    return df_out

def load_lda_vectorized_artifacts():
    """
    Load LDA vectorization outputs from disk (BoW + TF-IDF).

    Returns
    -------
    dict
        Dictionary containing:
        - X_bow: SciPy sparse matrix (documents x vocab)
        - X_tfidf: SciPy sparse matrix (documents x vocab)
        - bow_vocab: list[str] feature names for BoW columns
        - tfidf_vocab: list[str] feature names for TF-IDF columns
        - doc_ids: Optional[np.ndarray] aligned doc IDs (if saved)
        - texts: Optional[list[str]] aligned processed texts (if saved)

    Raises
    ------
    FileNotFoundError
        If required vectorized files are missing.
    """
    if not LDA_BOW_MATRIX_FILE.exists():
        raise FileNotFoundError(f"Missing BoW matrix at: {LDA_BOW_MATRIX_FILE}")
    if not LDA_TFIDF_MATRIX_FILE.exists():
        raise FileNotFoundError(f"Missing TF-IDF matrix at: {LDA_TFIDF_MATRIX_FILE}")
    if not LDA_BOW_FEATURE_NAMES_FILE.exists():
        raise FileNotFoundError(f"Missing BoW vocab at: {LDA_BOW_FEATURE_NAMES_FILE}")
    if not LDA_TFIDF_FEATURE_NAMES_FILE.exists():
        raise FileNotFoundError(f"Missing TF-IDF vocab at: {LDA_TFIDF_FEATURE_NAMES_FILE}")

    X_bow = load_npz(LDA_BOW_MATRIX_FILE)
    X_tfidf = load_npz(LDA_TFIDF_MATRIX_FILE)

    bow_vocab = joblib.load(LDA_BOW_FEATURE_NAMES_FILE)
    tfidf_vocab = joblib.load(LDA_TFIDF_FEATURE_NAMES_FILE)

    doc_ids = None
    if LDA_DOC_IDS_FILE.exists():
        doc_ids = np.load(LDA_DOC_IDS_FILE, allow_pickle=True)

    texts = None
    if LDA_PROCESSED_TEXT_FILE.exists():
        texts = joblib.load(LDA_PROCESSED_TEXT_FILE)

    # -------------------------
    # Sanity checks (alignment)
    # -------------------------
    n_docs = X_bow.shape[0]

    if X_tfidf.shape[0] != n_docs:
        raise ValueError(
            f"X_tfidf rows {X_tfidf.shape[0]} != X_bow rows {n_docs}"
        )

    if doc_ids is not None and len(doc_ids) != n_docs:
        raise ValueError(
            f"doc_ids length {len(doc_ids)} != X_bow rows {n_docs}"
        )

    if texts is not None and len(texts) != n_docs:
        raise ValueError(
            f"texts length {len(texts)} != X_bow rows {n_docs}"
        )

    return {
        "X_bow": X_bow,
        "X_tfidf": X_tfidf,
        "bow_vocab": bow_vocab,
        "tfidf_vocab": tfidf_vocab,
        "doc_ids": doc_ids,
        "texts": texts,
    }

def load_lda_bow():
    """
    Convenience loader: load only BoW artifacts for LDA.
    """
    artifacts = load_lda_vectorized_artifacts()
    return artifacts["X_bow"], artifacts["bow_vocab"], artifacts["doc_ids"]


def load_lda_tfidf():
    """
    Convenience loader: load only TF-IDF artifacts for LDA.
    """
    artifacts = load_lda_vectorized_artifacts()
    return artifacts["X_tfidf"], artifacts["tfidf_vocab"], artifacts["doc_ids"]

if __name__ == "__main__":
    # Quick manual tests: run `python -m src.data_loader` from project root
    df_raw = load_raw_complaints()
    print(df_raw.head())
    print(f"\nLoaded {len(df_raw)} rows from raw complaints file.")

    # If cleaned file exists, test it too
    try:
        df_clean = load_cleaned_complaints()
        print("\nCleaned file preview:")
        print(df_clean.head())
        print(f"\nLoaded {len(df_clean)} rows from cleaned complaints file.")

        df_lda = load_cleaned_for_lda()
        print("\nLDA-ready preview:")
        print(df_lda.head())
        print(f"\nLoaded {len(df_lda)} rows for LDA vectorization.")

        df_bt = load_cleaned_for_bertopic()
        print("\nBERTopic-ready preview:")
        print(df_bt.head())
        print(f"\nLoaded {len(df_bt)} rows for BERTopic.")
    except FileNotFoundError:
        print("\nCleaned complaints file not found yet (run cleaning.py first).")

    # If vectorized artifacts exist, test loading them too
    try:
        artifacts = load_lda_vectorized_artifacts()
        print("\nVectorized LDA artifacts preview:")
        print("BoW shape:", artifacts["X_bow"].shape)
        print("TF-IDF shape:", artifacts["X_tfidf"].shape)
        print("BoW vocab size:", len(artifacts["bow_vocab"]))
        print("TF-IDF vocab size:", len(artifacts["tfidf_vocab"]))
    except FileNotFoundError:
        print("\nVectorized artifacts not found yet (run vectorization.py first).")

