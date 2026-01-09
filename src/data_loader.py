"""
data_loader.py
--------------

Data loading utilities for Munich Open311 complaints topic modeling pipeline.

This module provides unified loaders for all data artifacts produced by the
topic modeling workflow. It handles loading raw data, cleaned texts, vectorized
matrices, trained models, and labeled topics.

Data Sources
------------
Raw Data:
  - data/raw/munich_open311_2020-01-01_to_2025-12-01.csv (original complaint descriptions)

Cleaned Data:
  - data/cleaned/cleaned_lda_berttopic.csv (LDA + BERTopic cleaned texts)

Vectorized Data (LDA only):
  - data/vectorized/lda/lda_bow_matrix.npz (BoW sparse matrix)
  - data/vectorized/lda/lda_tfidf_matrix.npz (TF-IDF sparse matrix)
  - data/vectorized/lda/lda_bow_feature_names.joblib (BoW vocabulary)
  - data/vectorized/lda/lda_tfidf_feature_names.joblib (TF-IDF vocabulary)

Topic Models:
  - data/topic_models/lda/ (trained LDA models, topic tables, doc-topics)
  - data/topic_models/bertopic/ (trained BERTopic model, topic info)

Labeled Topics (LLM-generated):
  - data/topic_models/lda/german_labeled/ (German topic labels)
  - data/topic_models/lda/english_labeled/ (English topic labels)
  - data/topic_models/bertopic/german_labeled/ (German topic labels)
  - data/topic_models/bertopic/english_labeled/ (English topic labels)

Loader Functions
----------------
Raw & Cleaned Data:
  - load_raw_complaints()
  - load_cleaned_complaints()
  - load_cleaned_for_lda()
  - load_cleaned_for_bertopic()

Vectorized Artifacts (LDA):
  - load_lda_vectorized_artifacts()
  - load_lda_bow()
  - load_lda_tfidf()

LDA Topic Models:
  - load_lda_bow_model()
  - load_lda_bow_topics()
  - load_lda_bow_doc_topics()
  - load_lda_bow_info()
  - load_lda_bow_k_sweep()
  - load_lda_tfidf_model()
  - load_lda_tfidf_topics()
  - load_lda_tfidf_doc_topics()
  - load_lda_tfidf_info()
  - load_lda_tfidf_k_sweep()

BERTopic Models:
  - load_bertopic_model()
  - load_bertopic_topic_info()
  - load_bertopic_doc_topics()

Labeled Topics:
  - load_lda_labeled_topics()
  - load_lda_labeled_doc_topics()
  - load_bertopic_labeled_topics()
  - load_bertopic_labeled_doc_topics()
  - load_bertopic_model_labeled()

Usage
-----
Load raw data:
    from src.data_loader import load_raw_complaints
    df = load_raw_complaints()

Load cleaned data for LDA:
    from src.data_loader import load_cleaned_for_lda
    df = load_cleaned_for_lda(keep_id=True)

Load vectorized artifacts:
    from src.data_loader import load_lda_vectorized_artifacts
    artifacts = load_lda_vectorized_artifacts()
    X_bow = artifacts["X_bow"]
    vocab = artifacts["bow_vocab"]

Load trained models:
    from src.data_loader import load_lda_bow_model, load_bertopic_model
    lda = load_lda_bow_model()
    bertopic = load_bertopic_model()

Load labeled topics:
    from src.data_loader import load_lda_labeled_topics
    topics_df = load_lda_labeled_topics(model_type='bow', language='de')

Notes
-----
- All loaders raise FileNotFoundError if required files are missing
- Loaders validate data alignment (e.g., X_bow rows == X_tfidf rows)
- Language-specific loaders support both 'de' (German) and 'en' (English)
- LDA vectorization uses only 'lda_description' column from cleaned file
- BERTopic uses 'bertopic_description' directly (no external vectorization)
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import load_npz
from gensim.models import LdaModel
from bertopic import BERTopic
from src.config import (
    RAW_COMPLAINTS_FILE,
    CLEANED_COMPLAINTS_FILE,
    LDA_BOW_MATRIX_FILE,
    LDA_TFIDF_MATRIX_FILE,
    LDA_BOW_FEATURE_NAMES_FILE,
    LDA_TFIDF_FEATURE_NAMES_FILE,
    LDA_DOC_IDS_FILE,
    LDA_PROCESSED_TEXT_FILE,
    LDA_VECTORIZATION_META_FILE,
    # LDA topic model artifact paths
    LDA_BOW_MODEL_FILE,
    LDA_BOW_TOPICS_FILE,
    LDA_BOW_DOC_TOPICS_FILE,
    LDA_BOW_INFO_FILE,
    LDA_BOW_K_SWEEP_FILE,
    LDA_TFIDF_MODEL_FILE,
    LDA_TFIDF_TOPICS_FILE,
    LDA_TFIDF_DOC_TOPICS_FILE,
    LDA_TFIDF_INFO_FILE,
    LDA_TFIDF_K_SWEEP_FILE,
    # BERTopic artifact paths
    BERTOPIC_MODEL_FILE,
    BERTOPIC_TOPIC_INFO_FILE,
    BERTOPIC_DOC_TOPICS_FILE,
    # Language-specific directories
    LDA_TOPIC_DIR_DE,
    LDA_TOPIC_DIR_EN,
    BERTOPIC_TOPIC_DIR_DE,
    BERTOPIC_TOPIC_DIR_EN,
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

    required_cols = ["description"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in complaints CSV: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

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

    meta = None
    if LDA_VECTORIZATION_META_FILE.exists():
        meta = joblib.load(LDA_VECTORIZATION_META_FILE)

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

    if meta is not None:
        required_keys = {"lowercase", "token_pattern", "ngram_range", "min_df", "max_df"}
        missing = required_keys - set(meta.keys())
        if missing:
            raise ValueError(f"LDA vectorization meta is missing keys: {sorted(missing)}")

    return {
        "X_bow": X_bow,
        "X_tfidf": X_tfidf,
        "bow_vocab": bow_vocab,
        "tfidf_vocab": tfidf_vocab,
        "doc_ids": doc_ids,
        "texts": texts,
        "meta": meta,
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


# =======================
# LDA Topic Model Loaders
# =======================

def load_lda_bow_model():
    """
    Load the trained BoW LDA model.

    Returns
    -------
    gensim.models.LdaModel
        Trained LDA model for BoW vectorization.

    Raises
    ------
    FileNotFoundError
        If model file is missing.
    """
    if not LDA_BOW_MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing BoW LDA model at: {LDA_BOW_MODEL_FILE}")

    return LdaModel.load(str(LDA_BOW_MODEL_FILE))


def load_lda_bow_topics():
    """
    Load BoW LDA topic-word distributions.

    Returns
    -------
    pd.DataFrame
        Topic-word distributions with columns: topic_id, word, probability.

    Raises
    ------
    FileNotFoundError
        If topics CSV is missing.
    """
    if not LDA_BOW_TOPICS_FILE.exists():
        raise FileNotFoundError(f"Missing BoW topics CSV at: {LDA_BOW_TOPICS_FILE}")

    return pd.read_csv(LDA_BOW_TOPICS_FILE)


def load_lda_bow_doc_topics():
    """
    Load BoW LDA document-topic distributions.

    Returns
    -------
    pd.DataFrame
        Document-topic distributions with topic probabilities per document.

    Raises
    ------
    FileNotFoundError
        If doc-topics CSV is missing.
    """
    if not LDA_BOW_DOC_TOPICS_FILE.exists():
        raise FileNotFoundError(f"Missing BoW doc-topics CSV at: {LDA_BOW_DOC_TOPICS_FILE}")

    return pd.read_csv(LDA_BOW_DOC_TOPICS_FILE)


def load_lda_bow_info():
    """
    Load BoW LDA model metadata.

    Returns
    -------
    dict
        Metadata including coherence, num_topics, n_docs, etc.

    Raises
    ------
    FileNotFoundError
        If info file is missing.
    """
    if not LDA_BOW_INFO_FILE.exists():
        raise FileNotFoundError(f"Missing BoW info file at: {LDA_BOW_INFO_FILE}")

    return joblib.load(LDA_BOW_INFO_FILE)


def load_lda_bow_k_sweep():
    """
    Load BoW LDA grid search results.

    Returns
    -------
    pd.DataFrame
        K-sweep results with columns: k, coherence, perplexity.

    Raises
    ------
    FileNotFoundError
        If K-sweep CSV is missing.
    """
    if not LDA_BOW_K_SWEEP_FILE.exists():
        raise FileNotFoundError(f"Missing BoW K-sweep CSV at: {LDA_BOW_K_SWEEP_FILE}")

    return pd.read_csv(LDA_BOW_K_SWEEP_FILE)


def load_lda_tfidf_model():
    """
    Load the trained TF-IDF LDA model.

    Returns
    -------
    gensim.models.LdaModel
        Trained LDA model for TF-IDF vectorization.

    Raises
    ------
    FileNotFoundError
        If model file is missing.
    """
    if not LDA_TFIDF_MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing TF-IDF LDA model at: {LDA_TFIDF_MODEL_FILE}")

    return LdaModel.load(str(LDA_TFIDF_MODEL_FILE))


def load_lda_tfidf_topics():
    """
    Load TF-IDF LDA topic-word distributions.

    Returns
    -------
    pd.DataFrame
        Topic-word distributions with columns: topic_id, word, probability.

    Raises
    ------
    FileNotFoundError
        If topics CSV is missing.
    """
    if not LDA_TFIDF_TOPICS_FILE.exists():
        raise FileNotFoundError(f"Missing TF-IDF topics CSV at: {LDA_TFIDF_TOPICS_FILE}")

    return pd.read_csv(LDA_TFIDF_TOPICS_FILE)


def load_lda_tfidf_doc_topics():
    """
    Load TF-IDF LDA document-topic distributions.

    Returns
    -------
    pd.DataFrame
        Document-topic distributions with topic probabilities per document.

    Raises
    ------
    FileNotFoundError
        If doc-topics CSV is missing.
    """
    if not LDA_TFIDF_DOC_TOPICS_FILE.exists():
        raise FileNotFoundError(f"Missing TF-IDF doc-topics CSV at: {LDA_TFIDF_DOC_TOPICS_FILE}")

    return pd.read_csv(LDA_TFIDF_DOC_TOPICS_FILE)


def load_lda_tfidf_info():
    """
    Load TF-IDF LDA model metadata.

    Returns
    -------
    dict
        Metadata including coherence, num_topics, n_docs, etc.

    Raises
    ------
    FileNotFoundError
        If info file is missing.
    """
    if not LDA_TFIDF_INFO_FILE.exists():
        raise FileNotFoundError(f"Missing TF-IDF info file at: {LDA_TFIDF_INFO_FILE}")

    return joblib.load(LDA_TFIDF_INFO_FILE)


def load_lda_tfidf_k_sweep():
    """
    Load TF-IDF LDA grid search results.

    Returns
    -------
    pd.DataFrame
        K-sweep results with columns: k, coherence, perplexity.

    Raises
    ------
    FileNotFoundError
        If K-sweep CSV is missing.
    """
    if not LDA_TFIDF_K_SWEEP_FILE.exists():
        raise FileNotFoundError(f"Missing TF-IDF K-sweep CSV at: {LDA_TFIDF_K_SWEEP_FILE}")

    return pd.read_csv(LDA_TFIDF_K_SWEEP_FILE)


# =======================
# BERTopic Model Loaders
# =======================

def load_bertopic_model():
    """
    Load the trained BERTopic model.

    Returns
    -------
    BERTopic
        Trained BERTopic model.

    Raises
    ------
    FileNotFoundError
        If model file is missing.
    """
    if not BERTOPIC_MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing BERTopic model at: {BERTOPIC_MODEL_FILE}")

    return BERTopic.load(str(BERTOPIC_MODEL_FILE))


def load_bertopic_topic_info():
    """
    Load BERTopic topic information table.

    Returns
    -------
    pd.DataFrame
        Topic info with columns: Topic, Count, Name, Representation, Representative_Docs.

    Raises
    ------
    FileNotFoundError
        If topic info CSV is missing.
    """
    if not BERTOPIC_TOPIC_INFO_FILE.exists():
        raise FileNotFoundError(f"Missing BERTopic topic info CSV at: {BERTOPIC_TOPIC_INFO_FILE}")

    return pd.read_csv(BERTOPIC_TOPIC_INFO_FILE)


def load_bertopic_doc_topics():
    """
    Load BERTopic document-topic assignments.

    Returns
    -------
    pd.DataFrame
        Document-topic assignments with doc IDs and assigned topics.

    Raises
    ------
    FileNotFoundError
        If doc-topics CSV is missing.
    """
    if not BERTOPIC_DOC_TOPICS_FILE.exists():
        raise FileNotFoundError(f"Missing BERTopic doc-topics CSV at: {BERTOPIC_DOC_TOPICS_FILE}")

    return pd.read_csv(BERTOPIC_DOC_TOPICS_FILE)


# ============================================
# Language-Specific Labeled Topic Loaders
# ============================================

def load_lda_labeled_topics(
    model_type: str = 'bow',
    language: str = 'de'
) -> pd.DataFrame:
    """
    Load LDA topic-word table with LLM-generated topic names.

    Parameters
    ----------
    model_type : str, default='bow'
        Type of LDA model: 'bow' or 'tfidf'
    language : str, default='de'
        Language for topic names: 'de' (German) or 'en' (English)

    Returns
    -------
    pd.DataFrame
        Topic-word distributions with additional 'topic_name' column.
        Columns: topic_id, word, weight, topic_name

    Raises
    ------
    FileNotFoundError
        If labeled topics CSV is missing.
    ValueError
        If invalid model_type or language is specified.
    """
    if model_type not in ['bow', 'tfidf']:
        raise ValueError(f"model_type must be 'bow' or 'tfidf', got: {model_type}")

    if language not in ['de', 'en']:
        raise ValueError(f"language must be 'de' or 'en', got: {language}")

    topic_dir = LDA_TOPIC_DIR_DE if language == 'de' else LDA_TOPIC_DIR_EN
    suffix = '' if language == 'de' else '_en'
    file_path = topic_dir / f"lda_{model_type}_topics_labeled{suffix}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Labeled LDA topics not found at: {file_path}\n"
            f"Run topic modeling with use_llm_labels=True first."
        )

    return pd.read_csv(file_path)


def load_lda_labeled_doc_topics(
    model_type: str = 'bow',
    language: str = 'de'
) -> pd.DataFrame:
    """
    Load LDA document-topic distributions with topic names.

    Parameters
    ----------
    model_type : str, default='bow'
        Type of LDA model: 'bow' or 'tfidf'
    language : str, default='de'
        Language for topic names: 'de' (German) or 'en' (English)

    Returns
    -------
    pd.DataFrame
        Document-topic distributions with topic names.
        Columns: doc_index, doc_id, topic_0...topic_K, dominant_topic, dominant_topic_name

    Raises
    ------
    FileNotFoundError
        If labeled doc-topics CSV is missing.
    """
    if model_type not in ['bow', 'tfidf']:
        raise ValueError(f"model_type must be 'bow' or 'tfidf', got: {model_type}")

    if language not in ['de', 'en']:
        raise ValueError(f"language must be 'de' or 'en', got: {language}")

    topic_dir = LDA_TOPIC_DIR_DE if language == 'de' else LDA_TOPIC_DIR_EN
    suffix = '' if language == 'de' else '_en'
    file_path = topic_dir / f"lda_{model_type}_doc_topics_labeled{suffix}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Labeled LDA doc-topics not found at: {file_path}\n"
            f"Run topic modeling with use_llm_labels=True first."
        )

    return pd.read_csv(file_path)


def load_bertopic_labeled_topics(language: str = 'de') -> pd.DataFrame:
    """
    Load BERTopic topic info with LLM-generated topic names.

    Parameters
    ----------
    language : str, default='de'
        Language for topic names: 'de' (German) or 'en' (English)

    Returns
    -------
    pd.DataFrame
        Topic info with LLM-generated names.
        Columns: Topic, Count, Name (LLM-generated), Representation, etc.

    Raises
    ------
    FileNotFoundError
        If labeled topics CSV is missing.
    ValueError
        If invalid language is specified.
    """
    if language not in ['de', 'en']:
        raise ValueError(f"language must be 'de' or 'en', got: {language}")

    topic_dir = BERTOPIC_TOPIC_DIR_DE if language == 'de' else BERTOPIC_TOPIC_DIR_EN
    suffix = '' if language == 'de' else '_en'
    file_path = topic_dir / f"bertopic_topic_info_labeled{suffix}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Labeled BERTopic topics not found at: {file_path}\n"
            f"Run topic modeling with use_llm_labels=True first."
        )

    return pd.read_csv(file_path)


def load_bertopic_labeled_doc_topics(language: str = 'de') -> pd.DataFrame:
    """
    Load BERTopic document-topic assignments with topic names.

    Parameters
    ----------
    language : str, default='de'
        Language for topic names: 'de' (German) or 'en' (English)

    Returns
    -------
    pd.DataFrame
        Document-topic assignments with topic names.
        Columns: doc_id, topic_id, topic_name, etc.

    Raises
    ------
    FileNotFoundError
        If labeled doc-topics CSV is missing.
    """
    if language not in ['de', 'en']:
        raise ValueError(f"language must be 'de' or 'en', got: {language}")

    topic_dir = BERTOPIC_TOPIC_DIR_DE if language == 'de' else BERTOPIC_TOPIC_DIR_EN
    suffix = '' if language == 'de' else '_en'
    file_path = topic_dir / f"bertopic_doc_topics_labeled{suffix}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Labeled BERTopic doc-topics not found at: {file_path}\n"
            f"Run topic modeling with use_llm_labels=True first."
        )

    return pd.read_csv(file_path)


def load_bertopic_model_labeled(language: str = 'de'):
    """
    Load trained BERTopic model from language-specific directory.

    Parameters
    ----------
    language : str, default='de'
        Language for model: 'de' (German) or 'en' (English)

    Returns
    -------
    BERTopic
        Trained BERTopic model with LLM-generated topic names.

    Raises
    ------
    FileNotFoundError
        If model is missing.
    """
    if language not in ['de', 'en']:
        raise ValueError(f"language must be 'de' or 'en', got: {language}")

    topic_dir = BERTOPIC_TOPIC_DIR_DE if language == 'de' else BERTOPIC_TOPIC_DIR_EN
    model_path = topic_dir / "bertopic_model"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Labeled BERTopic model not found at: {model_path}\n"
            f"Run topic modeling with use_llm_labels=True first."
        )

    return BERTopic.load(str(model_path))


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

