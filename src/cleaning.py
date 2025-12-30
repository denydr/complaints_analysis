"""
cleaning.py
-----------

This module implements the text cleaning pipeline for the Munich Open311
complaints dataset. It prepares the raw German complaint texts as inputs
for two different topic modeling approaches:

1. LDA (Latent Dirichlet Allocation)
2. BERTopic (transformer-based topic modeling)

The raw data are loaded via `data_loader.load_raw_complaints()` and the
cleaned outputs are saved as a single CSV file:

    - Path: config.CLEANED_COMPLAINTS_FILE
    - Default: data/cleaned/cleaned_lda_berttopic.csv

For each complaint, the following columns are produced:
-------------------------------------------------------
- service_request_id      (if present in the raw CSV and `keep_id=True`)
- description             (original raw German complaint text)
- lda_description         (cleaned, lemmatized text for LDA)
- bertopic_description    (lightly cleaned text for BERTopic)

Cleaning Strategies:
--------------------
- LDA:
  - Strong normalization and noise removal.
  - Removes URLs, emails, punctuation, HTML-like artifacts,
    and non-alphanumeric symbols.
  - Uses spaCy (German model) for tokenization and lemmatization.
  - Removes German stopwords and very short tokens.
  - Produces a space-separated string of lemmas, optimized for
    bag-of-words models like LDA.

- BERTopic:
  - Reuses the same robust regex-based noise removal as LDA for
    URLs/emails/special characters, but:
      * No spaCy lemmatization.
      * No stopword removal.
  - Keeps digits, letters (incl. German umlauts and ß), and spaces.
  - Drops junk single-letter alphabetic tokens (e.g. "p" left from "<p>").
  - Preserves more of the original sentence structure so that
    transformer-based embeddings (used in BERTopic) can exploit
    the contextual information.

Command-Line Usage:
-------------------
From the project root (with `src/` as a package):

    python -m src.cleaning

This will:
    - Load the raw complaints CSV.
    - Generate `lda_description` and `bertopic_description`.
    - Save the resulting DataFrame to CLEANED_COMPLAINTS_FILE.
"""

import re

import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS as GERMAN_STOP_WORDS

from src.config import CLEANED_COMPLAINTS_FILE
from src.data_loader import load_raw_complaints

# Additional stopwords specific to complaint-formalities / boilerplate
LDA_EXTRA_STOPWORDS = {
    "bitte", "danke", "vielen", "dank",
    "hallo", "guten", "tag",
    "sehr", "geehrt", "geehrte", "geehrter",
    "freundlich", "freundlichen", "grüße", "gruss", "gruß", "gruesse",
    "mit", "besten", "anbei", "bzgl", "bzw",
    "herr", "damen", "dame",
    # street / admin noise
    "str", "strasse", "straße", "nr", "nummer", "hausnummer",
}

# -------------------------
# spaCy & regex setup
# -------------------------

# Loading the spaCy model for German.
# Disable NER and dependency parser to speed up preprocessing,
# since LDA only needs tokenization + lemmatization.
nlp_de = spacy.load("de_core_news_sm", disable=["ner", "parser"])

# Simple regex patterns used by both pipelines for early noise removal.
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_for_lda(text: str, lowercase: bool = True) -> str:
    """
    LDA-oriented cleaning pipeline for German complaints.

    Chronology:
    1) Punctuation & special character removal (plus URLs/emails).
       - Uses a regex to keep only digits, letters (incl. German umlauts and ß),
         and spaces; drops HTML-like tags, and other symbols.
    2) spaCy (German) for tokenization, stopword removal, lemmatization.

    Parameters
    ----------
    text : str
        Raw complaint description (German).
    lowercase : bool, default=True
        If True, convert lemmas to lowercase to reduce sparsity in the
        bag-of-words representation.

    Returns
    -------
    str
        Cleaned, lemmatized text for LDA (space-separated tokens),
        suitable for vectorization (e.g. CountVectorizer/TF-IDF).
    """
    if not isinstance(text, str):
        text = str(text)

    # --- Step 1: Punctuation & special character removal (and noise) ---

    # Remove URLs and emails
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)

    # Keep digits, letters (incl. German umlauts and ß) and spaces
    # Remove other special characters / punctuation
    text = re.sub(r"[^0-9A-Za-zÄÖÜäöüß ]+", " ", text)

    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    if not text:
        return ""

    # --- Step 2: spaCy: tokenization, stopword removal, lemmatization ---

    doc = nlp_de(text)

    tokens = []
    for token in doc:
        # Skip spaces, punctuation, and pure digits
        if token.is_space or token.is_punct or token.is_digit:
            continue

        lemma = token.lemma_.strip()
        if not lemma:
            continue

        # Optional lowercasing of lemmas (helps LDA a lot)
        if lowercase:
            lemma = lemma.lower()

        if lemma in GERMAN_STOP_WORDS or lemma in LDA_EXTRA_STOPWORDS or len(lemma) < 3:
            continue

        tokens.append(lemma)

    return " ".join(tokens)


def clean_for_bertopic(text: str, lowercase: bool = True) -> str:
    """
    BERTopic-oriented cleaning pipeline for German complaints.

    This pipeline reuses the robust noise-removal from the LDA cleaner,
    but intentionally:
    - Does NOT use spaCy lemmatization.
    - Does NOT remove stopwords.

    The goal is to keep the text relatively natural so that transformer-based
    embeddings (used in BERTopic) can still exploit word order and local
    context, while removing obvious noise.

    Steps
    -----
    1) Remove URLs and emails using shared regexes.
    2) Keep only digits, letters (incl. German umlauts and ß) and spaces:
       -> This removes HTML tags like `<p>`, `<br>`, brackets, and
          other non-alphanumeric symbols.
    3) Normalize whitespace.
    4) Optional lowercasing.
    5) Drop junky single-letter alphabetic tokens (e.g. "p" from "<p>").

    Parameters
    ----------
    text : str
        Raw complaint description (German).
    lowercase : bool, default=True
        If True, lowercase the cleaned text.

    Returns
    -------
    str
        Lightly cleaned text suitable for BERTopic (sequence of tokens
        as a single string, preserving more of the original semantics
        than the LDA pipeline).
    """
    if not isinstance(text, str):
        text = str(text)

    # 1) Remove URLs and emails
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)

    # 2) Reuse LDA's regex: keep digits, letters (incl. umlauts, ß) and spaces.
    #    This also removes <p>, <br>, and other non-alphanumeric junk.
    text = re.sub(r"[^0-9A-Za-zÄÖÜäöüß ]+", " ", text)

    # 3) Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    if not text:
        return ""

    # 4) Optional lowercasing
    if lowercase:
        text = text.lower()

    # 5) Drop junky single-letter alphabetic tokens (e.g. 'p' left from <p>)
    tokens = []
    for tok in text.split():
        t = tok.strip()
        if not t:
            continue
        if len(t) == 1 and t.isalpha():
            continue
        tokens.append(t)

    return " ".join(tokens)


def preprocess_and_save_cleaned_complaints(
    id_column: str = "service_request_id",
    keep_id: bool = True,
) -> pd.DataFrame:
    """
    Run the LDA- and BERTopic-oriented cleaning pipelines on the full
    complaints dataset and save the result into the cleaned directory
    as 'cleaned_lda_berttopic.csv'.

    The cleaned CSV will typically contain:
    - service_request_id       (if present and keep_id=True)
    - description              (raw German complaint text)
    - lda_description          (cleaned text for LDA)
    - bertopic_description     (lightly cleaned text for BERTopic)

    This function acts as the main entry point for the cleaning step in
    the overall workflow:
    - Loads the raw data via `load_raw_complaints()`.
    - Applies `clean_for_lda` and `clean_for_bertopic` to each description.
    - Drops rows where `lda_description` becomes empty (only stopwords/noise).
    - Persists the cleaned DataFrame to disk.

    Parameters
    ----------
    id_column : str, default="service_request_id"
        Name of the ID column in the raw CSV (if available).
    keep_id : bool, default=True
        If True and the ID column exists, keep it in the cleaned output.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame that was saved to CLEANED_COMPLAINTS_FILE.
    """
    # 1. Load raw data
    df_raw = load_raw_complaints()

    # 2. Decide which columns to keep (always keep description)
    cols_to_keep = ["description"]
    if keep_id and id_column in df_raw.columns:
        cols_to_keep.insert(0, id_column)  # ID first, then description

    df = df_raw[cols_to_keep].copy()

    # 3) Create a safe base text column (prevents "nan" strings)
    desc = df["description"].fillna("").astype(str)

    # 4) Apply cleaning
    df["lda_description"] = desc.apply(clean_for_lda)
    df["bertopic_description"] = desc.apply(clean_for_bertopic)

    # 5) (optional) Fix NaNs BEFORE filtering
    df["lda_description"] = df["lda_description"].fillna("")
    df["bertopic_description"] = df["bertopic_description"].fillna("")

    # 6) Drop rows empty for BOTH pipelines (shared dataset)
    df = df[
        (df["lda_description"].str.strip() != "") &
        (df["bertopic_description"].str.strip() != "")
        ].reset_index(drop=True)

    # 7. Save to cleaned directory as cleaned_lda_berttopic.csv
    CLEANED_COMPLAINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_COMPLAINTS_FILE, index=False)

    print(f"Saved LDA- and BERTopic-cleaned complaints to: {CLEANED_COMPLAINTS_FILE}")
    print(f"Rows after cleaning: {len(df)}")

    return df


if __name__ == "__main__":
    # Allows: python -m src.cleaning
    preprocess_and_save_cleaned_complaints()