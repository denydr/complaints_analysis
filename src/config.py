"""
config.py
---------

Central configuration module for defining:
- Project root
- Directory paths for raw, cleaned, and vectorized data
- Standard file names for key datasets

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
CLEANED_LDA_FILE = CLEANED_DIR / "cleaned_lda.csv" # cleaned complaints for LDA input

#TODO: Vectorized representations (you can refine these in vectorization.py)
TFIDF_MATRIX_FILE = VECTORIZED_DIR / ""
BOW_MATRIX_FILE = VECTORIZED_DIR / ""

# =======================
# Ensure Required Directories Exist
# =======================

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
VECTORIZED_DIR.mkdir(parents=True, exist_ok=True)