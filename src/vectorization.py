"""
vectorization.py
----------------

This module creates numerical representations (vectorizations) of the cleaned
German complaint texts for LDA topic modeling.

Two vectorization techniques are implemented for comparison:
1) Bag-of-Words (CountVectorizer)  -> term frequency counts
2) TF-IDF (TfidfVectorizer)        -> term frequency weighted by inverse document frequency

Input:
------
- Uses ONLY the 'lda_description' column from the cleaned dataset
  (loaded via data_loader.load_cleaned_for_lda()).

Outputs (saved under data/vectorized/lda/):
------------------------------------------
- BoW matrix as .npz
- TF-IDF matrix as .npz
- Feature names (vocabulary) for BoW as .joblib
- Feature names (vocabulary) for TF-IDF as .joblib
- Optional doc IDs (service_request_id) aligned with matrix rows as .npy
- Optional processed texts aligned with matrix rows as .joblib

Why feature names matter:
------------------------
The feature names allow for interpreting LDA topics later, because LDA returns
topic-word distributions over matrix columns. The column->token mapping are needed
to print the top words per topic.

Command-Line Usage:
-------------------
Run vectorization after cleaning:
    python -m src.vectorization
"""

import joblib
import numpy as np
from scipy.sparse import save_npz

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import (
    LDA_BOW_MATRIX_FILE,
    LDA_TFIDF_MATRIX_FILE,
    LDA_BOW_FEATURE_NAMES_FILE,
    LDA_TFIDF_FEATURE_NAMES_FILE,
    LDA_DOC_IDS_FILE,
    LDA_PROCESSED_TEXT_FILE,
)
from src.data_loader import load_cleaned_for_lda


def vectorize_for_lda_and_save(
    id_column: str = "service_request_id",
    keep_id: bool = True,
    min_df: int = 2,
    max_df: float = 0.85,
    ngram_range: tuple = (1, 1),
) -> None:
    """
    Vectorize the cleaned LDA texts with two techniques (BoW + TF-IDF),
    save the matrices and vocabularies to disk.

    Parameters
    ----------
    id_column : str
        Name of the document ID column (if present).
    keep_id : bool
        If True and id_column exists, save doc IDs aligned with matrix rows.
    min_df : int
        Ignore terms that appear in fewer than min_df documents.
    max_df : float
        Ignore terms that appear in more than max_df fraction of documents.
    ngram_range : tuple
        Use unigrams (1,1) by default. Can test (1,2) later if needed.
    """
    # 1) Load cleaned LDA texts (and optional IDs)
    df = load_cleaned_for_lda(id_column=id_column, keep_id=keep_id)

    texts = df["lda_description"].astype(str).tolist()

    # Save processed texts for reproducibility/debugging
    joblib.dump(texts, LDA_PROCESSED_TEXT_FILE)

    # Save document IDs if available
    if keep_id and id_column in df.columns:
        doc_ids = df[id_column].to_numpy()
        np.save(LDA_DOC_IDS_FILE, doc_ids)

    # 2) Create vectorizers with consistent settings for fair comparison
    # Important:
    # - lowercase=False because lda_description is already lowercased in cleaning
    # - token_pattern allows German umlauts/ß + digits  (punctuation is already cleaned)
    token_pattern = r"(?u)\b\w+\b"

    bow_vectorizer = CountVectorizer(
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
    )

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        token_pattern=token_pattern,
    )

    # 3) Fit-transform (same texts -> comparable document basis)
    X_bow = bow_vectorizer.fit_transform(texts)
    X_tfidf = tfidf_vectorizer.fit_transform(texts)

    # 3.1) Drop documents that became empty after vectorizer filtering (all-zero rows)
    bow_nonzero = np.asarray(X_bow.sum(axis=1)).ravel() > 0
    tfidf_nonzero = np.asarray(X_tfidf.sum(axis=1)).ravel() > 0
    keep_mask = bow_nonzero & tfidf_nonzero

    dropped = int((~keep_mask).sum())
    if dropped > 0:
        print(f"Dropping {dropped} documents that became empty after vectorization filters.")
        X_bow = X_bow[keep_mask]
        X_tfidf = X_tfidf[keep_mask]
        texts = list(np.array(texts)[keep_mask])

        if keep_id and id_column in df.columns:
            doc_ids = np.array(doc_ids)[keep_mask]
            np.save(LDA_DOC_IDS_FILE, doc_ids)

        # overwrite processed texts so they stay aligned
        joblib.dump(texts, LDA_PROCESSED_TEXT_FILE)

    # 4) Save sparse matrices as .npz
    save_npz(LDA_BOW_MATRIX_FILE, X_bow)
    save_npz(LDA_TFIDF_MATRIX_FILE, X_tfidf)

    # 5) Save feature names (vocabularies) separately
    bow_feature_names = bow_vectorizer.get_feature_names_out().tolist()
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out().tolist()

    joblib.dump(bow_feature_names, LDA_BOW_FEATURE_NAMES_FILE)
    joblib.dump(tfidf_feature_names, LDA_TFIDF_FEATURE_NAMES_FILE)

    # 6) Print basic stats for immediate comparison
    print("✅ Vectorization complete.")
    print(f"Documents: {len(texts)}")
    print(f"BoW matrix shape:   {X_bow.shape}")
    print(f"TF-IDF matrix shape:{X_tfidf.shape}")
    print(f"BoW vocab size:     {len(bow_feature_names)}")
    print(f"TF-IDF vocab size:  {len(tfidf_feature_names)}")
    print(f"Saved BoW matrix to:   {LDA_BOW_MATRIX_FILE}")
    print(f"Saved TF-IDF matrix to:{LDA_TFIDF_MATRIX_FILE}")
    print(f"Saved BoW vocab to:    {LDA_BOW_FEATURE_NAMES_FILE}")
    print(f"Saved TF-IDF vocab to: {LDA_TFIDF_FEATURE_NAMES_FILE}")

    if keep_id and id_column in df.columns:
        print(f"Saved doc IDs to:      {LDA_DOC_IDS_FILE}")

    print(f"Saved processed texts to:{LDA_PROCESSED_TEXT_FILE}")


if __name__ == "__main__":
    # Allows: python -m src.vectorization
    vectorize_for_lda_and_save()