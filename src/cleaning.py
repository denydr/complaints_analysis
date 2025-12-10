"""
cleaning.py
---------
#TODO for cleaning first it is logical to conduct the punctuation and special charachter removal

#TODO spacy utilization for stopword removal
#TODO then TOKENIZATION - and after that stopword removal and lemmatization

#TODO Investigate the necessary cleaning steps for german language complaints - keep in mind
#TODO that most probably the stopwords chould not be removed if they are preserving vital context,
#TODO e.g., not or sth. like this for the complaints

"""

import re
from typing import Optional  # currently not used, but fine to keep for later

import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS as GERMAN_STOP_WORDS

from config import CLEANED_LDA_FILE
from data_loader import load_raw_complaints

# -------------------------
# spaCy & regex setup
# -------------------------

# Loading the spaCy model for German.
# Disable NER and dependency parser to speed up preprocessing,
# since LDA only needs tokenization + lemmatization.
nlp_de = spacy.load("de_core_news_sm", disable=["ner", "parser"])

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_for_lda(text: str, lowercase: bool = True) -> str:
    """
    LDA-oriented cleaning pipeline for German complaints.

    Chronology:
    1) Punctuation & special character removal (plus URLs/emails).
    2) spaCy (German) for tokenization, stopword removal, lemmatization.

    Parameters
    ----------
    text : str
        Raw complaint description (German).
    lowercase : bool
        If True, convert lemmas to lowercase to reduce sparsity.

    Returns
    -------
    str
        Cleaned, lemmatized text for LDA (space-separated tokens).
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

        # Skip German stopwords and very short tokens (e.g. "im", "am")
        if lemma in GERMAN_STOP_WORDS or len(lemma) < 3:
            continue

        tokens.append(lemma)

    return " ".join(tokens)


def preprocess_for_lda_and_save(
    id_column: str = "service_request_id",
    keep_id: bool = True,
) -> pd.DataFrame:
    """
    Run the LDA-oriented cleaning pipeline on the full complaints dataset
    and save the result into the cleaned directory as 'cleaned_lda.csv'.

    The cleaned CSV will typically contain:
    - service_request_id (if present and keep_id=True)
    - description        (raw German complaint text)
    - lda_description    (cleaned, lemmatized text for LDA)

    Parameters
    ----------
    id_column : str
        Name of the ID column in the raw CSV (if available).
    keep_id : bool
        If True and the ID column exists, keep it in the cleaned output.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame that was saved.
    """
    # 1. Load raw data
    df_raw = load_raw_complaints()

    # 2. Decide which columns to keep (always keep description)
    cols_to_keep = ["description"]
    if keep_id and id_column in df_raw.columns:
        cols_to_keep.insert(0, id_column)  # ID first, then description

    df = df_raw[cols_to_keep].copy()

    # 3. Apply LDA cleaning pipeline to the description
    df["lda_description"] = df["description"].astype(str).apply(clean_for_lda)

    # 4. Drop rows where lda_description is empty (only stopwords/noise)
    df = df[df["lda_description"].str.strip().ne("")].reset_index(drop=True)

    # 5. Save to cleaned directory as cleaned_lda.csv
    CLEANED_LDA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_LDA_FILE, index=False)

    print(f"Saved LDA-cleaned complaints to: {CLEANED_LDA_FILE}")
    print(f"Rows after cleaning: {len(df)}")

    return df


if __name__ == "__main__":
    # Allows: python -m src.cleaning
    preprocess_for_lda_and_save()