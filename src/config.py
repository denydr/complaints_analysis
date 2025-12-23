"""
config.py
---------

Central configuration module for defining:
- Project root
- Directory paths for raw, cleaned, and vectorized data
- Standard file names for key datasets
- Shared topic-modeling settings (random seeds, embedding model name)
- LLM configuration for topic labeling
- Language-specific output directories (German/English)

No execution logic beyond creating directories.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# LDA language-specific subdirectories for LLM-labeled artifacts
# German labeled directory: contains German topic artifacts with LLM-generated topic names
LDA_TOPIC_DIR_DE = LDA_TOPIC_DIR / "german_labeled"
# English labeled directory: contains English-translated topic artifacts with LLM-generated topic names
LDA_TOPIC_DIR_EN = LDA_TOPIC_DIR / "english_labeled"

# BERTopic outputs (saved model, topic info, doc-topic assignments)
BERTOPIC_TOPIC_DIR = TOPIC_MODELS_DIR / "bertopic"

# BERTopic language-specific subdirectories for LLM-labeled artifacts
# German labeled directory: contains German topic artifacts with LLM-generated topic names
BERTOPIC_TOPIC_DIR_DE = BERTOPIC_TOPIC_DIR / "german_labeled"
# English labeled directory: contains English-translated topic artifacts with LLM-generated topic names
BERTOPIC_TOPIC_DIR_EN = BERTOPIC_TOPIC_DIR / "english_labeled"

# =======================
# Topic Model Artifact Paths
# =======================

# LDA BoW (Bag-of-Words) model artifacts
# Trained Gensim LDA model for BoW vectorization
# Loaded with gensim.models.LdaModel.load()
LDA_BOW_MODEL_FILE = LDA_TOPIC_DIR / "lda_bow_model.gensim"

# Topic-word distributions for BoW LDA
# CSV with columns: topic_id, word, probability
LDA_BOW_TOPICS_FILE = LDA_TOPIC_DIR / "lda_bow_topics.csv"

# Document-topic distributions for BoW LDA
# CSV with columns: doc_id, topic_0, topic_1, ..., topic_K, dominant_topic
LDA_BOW_DOC_TOPICS_FILE = LDA_TOPIC_DIR / "lda_bow_doc_topic_distributions.csv"

# Model metadata for BoW LDA (coherence, num_topics, etc.)
# Dictionary stored with joblib.dump()
LDA_BOW_INFO_FILE = LDA_TOPIC_DIR / "lda_bow_info.joblib"

# Grid search results for BoW LDA K selection
# CSV with columns: k, coherence, perplexity
LDA_BOW_K_SWEEP_FILE = LDA_TOPIC_DIR / "lda_bow_k_sweep.csv"

# LDA TF-IDF model artifacts (same structure as BoW)
LDA_TFIDF_MODEL_FILE = LDA_TOPIC_DIR / "lda_tfidf_model.gensim"
LDA_TFIDF_TOPICS_FILE = LDA_TOPIC_DIR / "lda_tfidf_topics.csv"
LDA_TFIDF_DOC_TOPICS_FILE = LDA_TOPIC_DIR / "lda_tfidf_doc_topic_distributions.csv"
LDA_TFIDF_INFO_FILE = LDA_TOPIC_DIR / "lda_tfidf_info.joblib"
LDA_TFIDF_K_SWEEP_FILE = LDA_TOPIC_DIR / "lda_tfidf_k_sweep.csv"

# BERTopic model artifacts
# Trained BERTopic model saved with .save()
# Loaded with BERTopic.load()
BERTOPIC_MODEL_FILE = BERTOPIC_TOPIC_DIR / "bertopic_model"

# Topic information table for BERTopic
# CSV with columns: Topic, Count, Name, Representation, Representative_Docs
BERTOPIC_TOPIC_INFO_FILE = BERTOPIC_TOPIC_DIR / "bertopic_topic_info.csv"

# Document-topic assignments for BERTopic
# CSV with columns: doc_id, topic, topic_name
BERTOPIC_DOC_TOPICS_FILE = BERTOPIC_TOPIC_DIR / "bertopic_doc_topics.csv"

# =======================
# Visualization Output Directory
# =======================

# Base directory for all visualization outputs and analysis results
RESULTS_DIR = PROJECT_ROOT / "results"

# Language-specific results subdirectories for visualizations with LLM-labeled topics
# German labeled directory: visualizations with German LLM-generated topic names (files 05, 07, 09-14)
RESULTS_DIR_DE = RESULTS_DIR / "german_labeled"
# English labeled directory: visualizations with English LLM-generated topic names (files 05, 07, 09-14)
RESULTS_DIR_EN = RESULTS_DIR / "english_labeled"

# =======================
# LLM Configuration for Topic Labeling
# =======================

# OpenAI API key for LLM-based topic labeling
# Should be set as environment variable: export OPENAI_API_KEY="sk-..."
# Falls back to empty string if not set (will raise error when attempting to use LLM features)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM model to use for generating topic labels
# gpt-4o-mini: Cost-effective model, suitable for topic labeling tasks
# Alternative: "gpt-4o" for higher quality (but more expensive)
LLM_MODEL = "gpt-4o-mini"

# Number of top words to include in LLM prompts for topic labeling
LLM_NUM_KEYWORDS = 10

# Number of representative documents to include in LLM prompts for context
# BERTopic: uses built-in representative documents
# LDA: uses documents with highest topic probability
# Increased from 2 to 6 for better pattern recognition and reduced overgeneralization
LLM_NUM_REPRESENTATIVE_DOCS = 6

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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Language-specific subdirectories (created on demand when LLM labeling is enabled)
LDA_TOPIC_DIR_DE.mkdir(parents=True, exist_ok=True)
LDA_TOPIC_DIR_EN.mkdir(parents=True, exist_ok=True)
BERTOPIC_TOPIC_DIR_DE.mkdir(parents=True, exist_ok=True)
BERTOPIC_TOPIC_DIR_EN.mkdir(parents=True, exist_ok=True)
RESULTS_DIR_DE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR_EN.mkdir(parents=True, exist_ok=True)