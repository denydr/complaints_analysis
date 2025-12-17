"""
config.py
---------

Central configuration module for defining:
- Project root
- Directory paths for raw, cleaned, and vectorized data
- Standard file names for key datasets
- Shared topic-modeling settings (random seeds, embedding model name)

No execution logic beyond creating directories.
"""
from pathlib import Path

# =======================
# Project & Data Directories
# =======================

# src/ is one level below the project root, so we go one level up
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"

RAW_DIR = DATA_ROOT / "raw"
CLEANED_DIR = DATA_ROOT / "cleaned"
VECTORIZED_DIR = DATA_ROOT / "vectorized"

# =======================
# File Paths
# =======================

RAW_COMPLAINTS_FILE = RAW_DIR / "munich_open311_2020-01-01_to_2025-12-01.csv"

# Cleaned complaints (after cleaning.py)
CLEANED_COMPLAINTS_FILE = CLEANED_DIR / "cleaned_lda_berttopic.csv" # cleaned complaints for LDA and BERTopic input

# ============================================================
# Vectorization Output Paths for LDA Topic Modeling
# ============================================================

# Base directory for all LDA-related vectorized artifacts
LDA_VECTORIZED_DIR = VECTORIZED_DIR / "lda"

# Bag-of-Words (CountVectorizer) matrix for LDA
# Stored as a SciPy sparse matrix (.npz) using scipy.sparse.save_npz()
# Shape: (n_documents, n_features)
LDA_BOW_MATRIX_FILE = LDA_VECTORIZED_DIR / "lda_bow_matrix.npz"

# TF-IDF matrix for LDA (alternative input for comparison)
# Also stored as a SciPy sparse matrix (.npz)
# Shape: (n_documents, n_features)
LDA_TFIDF_MATRIX_FILE = LDA_VECTORIZED_DIR / "lda_tfidf_matrix.npz"

# Vocabulary (feature names) produced by the vectorizers
# A Python list of tokens, stored with joblib.dump().
# Required to interpret LDA topics (e.g., top 10 words per topic).
# The index of each token corresponds to the column index in the LDA matrices
# Vocabulary for Bag-of-Words (CountVectorizer)
LDA_BOW_FEATURE_NAMES_FILE = LDA_VECTORIZED_DIR / "lda_bow_feature_names.joblib"
# Vocabulary for TF-IDF (TfidfVectorizer)
LDA_TFIDF_FEATURE_NAMES_FILE = LDA_VECTORIZED_DIR / "lda_tfidf_feature_names.joblib"

# Optional: Document IDs aligned with the rows of the LDA matrices
# Typically the service_request_id column from the raw dataset
# Stored as a NumPy array (.npy) so topics can be mapped back to complaints
LDA_DOC_IDS_FILE = LDA_VECTORIZED_DIR / "lda_doc_ids.npy"

# Optional: Cleaned text used as input for LDA vectorization
# Corresponds exactly to the rows of the LDA matrices
# Stored as a list of strings via joblib.dump()
LDA_PROCESSED_TEXT_FILE = LDA_VECTORIZED_DIR / "lda_processed_texts.joblib"

# ============================================================
# Topic Modeling Output Paths
# ============================================================

# Base directory for all topic-modeling artifacts
TOPIC_MODELS_DIR = DATA_ROOT / "topic_models"

# LDA outputs (models, topic tables, coherence logs, etc.)
LDA_TOPIC_DIR = TOPIC_MODELS_DIR / "lda"

# BERTopic outputs (saved model, topic info, doc-topic assignments)
BERTOPIC_TOPIC_DIR = TOPIC_MODELS_DIR / "bertopic"

# =======================
# Topic Modeling Settings
# =======================

# Shared random seed for reproducibility across models that support it
RANDOM_STATE = 42

# LDA reproducibility (used later in topic_modeling.py)
LDA_RANDOM_STATE = RANDOM_STATE

# BERTopic settings (used later in topic_modeling.py)
# - Model will run on CPU by default; device is explicit for clarity.
# - UMAP is stochastic -> fix random state for reproducible topic runs.
BERTOPIC_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BERTOPIC_DEVICE = "cpu"
BERTOPIC_UMAP_RANDOM_STATE = RANDOM_STATE

# =======================
# Ensure Required Directories Exist
# =======================

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
VECTORIZED_DIR.mkdir(parents=True, exist_ok=True)
LDA_VECTORIZED_DIR.mkdir(parents=True, exist_ok=True)
TOPIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LDA_TOPIC_DIR.mkdir(parents=True, exist_ok=True)
BERTOPIC_TOPIC_DIR.mkdir(parents=True, exist_ok=True)