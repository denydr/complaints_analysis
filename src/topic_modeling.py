"""
topic_modeling.py
-----------------

Runs topic modeling on the cleaned Munich Open311 complaints dataset.

This module supports **three topic modeling pipelines**:

Pipelines
---------
1) LDA (Gensim) using Bag-of-Words (BoW) matrix
2) LDA (Gensim) using TF-IDF matrix (comparison run)
3) BERTopic using sentence-transformer embeddings on lightly cleaned text


Inputs
------
- LDA pipelines use vectorized artifacts produced by `vectorization.py`:
    - BoW / TF-IDF matrices (.npz)
    - vocabularies
    - optional document IDs
    - processed LDA texts (for coherence calculation)

- BERTopic uses:
    - `bertopic_description` loaded directly from the cleaned CSV


Outputs
-------
- LDA outputs (saved under `LDA_TOPIC_DIR/`):
    - trained Gensim models (.gensim)
    - topic-word tables (.csv)
    - per-document topic distributions (.csv)
    - coherence scores + metadata (.joblib)

- BERTopic outputs (saved under `BERTOPIC_TOPIC_DIR/`):
    - trained BERTopic model
    - topic summary table (`bertopic_topic_info.csv`)
    - per-document topic assignments (`bertopic_doc_topics.csv`)


How to Run
----------
1) Run BOTH LDA and BERTopic (default behavior):

    python -m src.topic_modeling

2) Run ONLY LDA (BoW + TF-IDF)
To run only the LDA pipelines:

This will:
- Train LDA on BoW
- Train LDA on TF-IDF
- Save LDA outputs
- Skip BERTopic entirely

3) Run ONLY BERTopic
To run only the BERTopic pipeline:

    python -m src.topic_modeling --bertopic

This will:
- Train BERTopic on `bertopic_description`
- Save BERTopic outputs
- Skip all LDA processing

4) Run BOTH explicitly (optional)
You may also run both pipelines explicitly:

    python -m src.topic_modeling --lda --bertopic

Notes
-----
- BERTopic performs its own internal vectorization for topic representations;
  stopwords removed there affect **topic labels only**, not embeddings.
- Random seeds for LDA and UMAP are fixed via `config.py` for reproducibility.
"""

from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd

from gensim import matutils
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
import spacy

from src.config import (
    LDA_RANDOM_STATE,
    BERTOPIC_EMBEDDING_MODEL,
    BERTOPIC_DEVICE,
    BERTOPIC_UMAP_RANDOM_STATE,
    LDA_TOPIC_DIR,
    BERTOPIC_TOPIC_DIR,
)
from src.data_loader import (
    load_lda_vectorized_artifacts,
    load_cleaned_for_bertopic,
)

# -------------------------
# Helpers
# -------------------------
def _save_topic_table(topics, out_csv: Path) -> None:
    """
    Save a topic-word table to CSV.

    Parameters
    ----------
    topics : list[tuple[int, list[tuple[str, float]]]]
        Topics in the format returned by Gensim when calling:
        `lda.show_topics(formatted=False)`.
        Example:
            [(0, [("word", 0.1), ...]), (1, [("word", 0.08), ...]), ...]
    out_csv : Path
        Output path for the CSV file.

    Returns
    -------
    None
        Writes a CSV with columns: topic_id, word, weight.
    """
    rows = []
    for topic_id, word_probs in topics:
        for word, prob in word_probs:
            rows.append({"topic_id": topic_id, "word": word, "weight": float(prob)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _save_doc_topic_distributions(
    lda: LdaModel,
    corpus,
    doc_ids,
    out_csv: Path
) -> None:
    """
    Save per-document topic distributions (gamma) to CSV.

    This is useful for:
    - Topic prevalence plots
    - Comparing BoW vs TF-IDF dominance patterns
    - Mapping topic assignments back to specific complaints (via doc_id)

    Parameters
    ----------
    lda : gensim.models.LdaModel
        Trained Gensim LDA model.
    corpus : iterable
        Gensim corpus in BoW format (as produced by matutils.Sparse2Corpus).
    doc_ids : array-like or None
        Optional document identifiers aligned with the corpus order.
        If None, only doc_index is saved.
    out_csv : Path
        Output path for the CSV file.

    Returns
    -------
    None
        Writes a CSV with columns:
        - doc_index
        - doc_id (if provided)
        - topic_0 ... topic_{k-1}
        - dominant_topic
    """
    num_topics = lda.num_topics

    rows = []
    for i, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        probs = np.zeros(num_topics, dtype=float)
        for topic_id, prob in dist:
            probs[int(topic_id)] = float(prob)

        dominant = int(np.argmax(probs))
        row = {"doc_index": i, "dominant_topic": dominant}
        if doc_ids is not None:
            row["doc_id"] = doc_ids[i]
        for t in range(num_topics):
            row[f"topic_{t}"] = float(probs[t])
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _train_gensim_lda(
    X_sparse,
    vocab: list[str],
    texts_tokens: list[list[str]],
    num_topics: int,
    passes: int,
    iterations: int,
    random_state: int,
    out_prefix: str,
    doc_ids=None,
):
    """
    Train a Gensim LDA model from a SciPy sparse matrix and persist artifacts.

    The function:
    - Converts SciPy sparse matrices to a Gensim corpus
    - Trains a Gensim LdaModel
    - Saves:
        - the trained model
        - topic-word table (CSV)
        - per-document topic distributions (CSV)
        - coherence + training metadata (joblib)

    Parameters
    ----------
    X_sparse : scipy.sparse matrix
        Document-term matrix with shape (n_docs, n_terms).
        Can be BoW counts or TF-IDF weights (for comparison).
    vocab : list[str]
        Vocabulary aligned with the columns of X_sparse.
        Column j corresponds to vocab[j].
    texts_tokens : list[list[str]]
        Tokenized documents used for coherence calculation.
        Must be aligned to the document order in X_sparse.
    num_topics : int
        Number of topics (K) to learn.
    passes : int
        Number of passes through the corpus (higher can improve convergence).
    iterations : int
        Maximum number of iterations per pass.
    random_state : int
        Random seed for reproducibility.
    out_prefix : str
        Prefix used for output filenames (e.g., "lda_bow" or "lda_tfidf").
    doc_ids : array-like or None, default=None
        Optional document IDs aligned to rows of X_sparse. If provided, saved in the
        per-document topic distribution file.

    Returns
    -------
    tuple
        (lda, coherence_c_v)
        - lda : gensim.models.LdaModel
            Trained model.
        - coherence_c_v : float
            c_v topic coherence score.
    """
    # Convert SciPy sparse -> gensim corpus
    corpus = matutils.Sparse2Corpus(X_sparse, documents_columns=False)

    # Gensim expects id2word mapping
    id2word = dict(enumerate(vocab))

    lda = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        random_state=random_state,
        eval_every=None,
        alpha="auto",
        eta="auto",
        chunksize=100,
    )

    # Save model
    model_path = LDA_TOPIC_DIR / f"{out_prefix}_model.gensim"
    lda.save(str(model_path))

    # Save topic table (top words)
    topics = lda.show_topics(num_topics=num_topics, num_words=15, formatted=False)
    _save_topic_table(topics, LDA_TOPIC_DIR / f"{out_prefix}_topics.csv")

    # Coherence (c_v) — align tokens to the trained vocabulary
    vocab_set = set(vocab)
    texts_tokens_aligned = [[tok for tok in doc if tok in vocab_set] for doc in texts_tokens]

    from gensim.corpora import Dictionary
    dictionary = Dictionary(texts_tokens_aligned)

    coherence_model = CoherenceModel(
        model=lda,
        texts=texts_tokens_aligned,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence = float(coherence_model.get_coherence())

    # Save per-doc topic distributions
    _save_doc_topic_distributions(
        lda=lda,
        corpus=corpus,
        doc_ids=doc_ids,
        out_csv=LDA_TOPIC_DIR / f"{out_prefix}_doc_topic_distributions.csv",
    )

    # Log + save info
    info = {
        "out_prefix": out_prefix,
        "num_topics": num_topics,
        "passes": passes,
        "iterations": iterations,
        "coherence_c_v": coherence,
        "n_docs": int(X_sparse.shape[0]),
        "n_terms": int(X_sparse.shape[1]),
        "matrix_dtype": str(X_sparse.dtype),
        "random_state": int(random_state),
    }
    joblib.dump(info, LDA_TOPIC_DIR / f"{out_prefix}_info.joblib")

    print(f"✅ LDA trained: {out_prefix}")
    print(f"   coherence (c_v): {coherence:.4f}")
    print(f"   saved model: {model_path}")

    return lda, coherence


# -------------------------
# Main pipelines
# -------------------------
def run_lda_pipelines(
    num_topics: int = 5,
    passes: int = 30,
    iterations: int = 250,
) -> None:
    """
    Train and compare two LDA models:
    1) LDA trained on BoW matrix
    2) LDA trained on TF-IDF matrix (comparison run)

    Parameters
    ----------
    num_topics : int, default=5
        Number of topics to learn for each model.
    passes : int, default=30
        Number of passes through the corpus.
    iterations : int, default=250
        Number of iterations per pass.

    Returns
    -------
    None
        Saves LDA artifacts under LDA_TOPIC_DIR and prints coherence comparison.
    """
    artifacts = load_lda_vectorized_artifacts()

    X_bow = artifacts["X_bow"]
    X_tfidf = artifacts["X_tfidf"]
    bow_vocab = artifacts["bow_vocab"]
    tfidf_vocab = artifacts["tfidf_vocab"]
    doc_ids = artifacts.get("doc_ids", None)

    texts = artifacts.get("texts", None)
    if texts is None:
        raise ValueError(
            "Missing LDA_PROCESSED_TEXT_FILE. Re-run vectorization.py with saving enabled."
        )
    texts_tokens = [t.split() for t in texts]

    _, coh_bow = _train_gensim_lda(
        X_sparse=X_bow,
        vocab=bow_vocab,
        texts_tokens=texts_tokens,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        random_state=LDA_RANDOM_STATE,
        out_prefix="lda_bow",
        doc_ids=doc_ids,
    )

    _, coh_tfidf = _train_gensim_lda(
        X_sparse=X_tfidf,
        vocab=tfidf_vocab,
        texts_tokens=texts_tokens,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        random_state=LDA_RANDOM_STATE,
        out_prefix="lda_tfidf",
        doc_ids=doc_ids,
    )

    print("\n=== LDA Comparison ===")
    print(f"BoW   coherence (c_v):  {coh_bow:.4f}")
    print(f"TF-IDF coherence (c_v): {coh_tfidf:.4f}")

def run_bertopic_pipeline(
    min_topic_size: int = 40,
    nr_topics: int | str | None = None,
    n_neighbors: int = 30,
    min_samples: int = 1,
) -> None:
    """
    Train a BERTopic model on lightly cleaned German complaint texts.

    Implements unsupervised topic discovery using transformer-based sentence
    embeddings (paraphrase-multilingual-MiniLM-L12-v2), UMAP dimensionality
    reduction, and HDBSCAN density-based clustering. Optimized for small-to-
    medium datasets (N ≈ 900 documents).

    The function:
    - Generates semantic embeddings using SentenceTransformers
    - Reduces dimensionality via UMAP (n_neighbors, n_components, min_dist)
    - Clusters documents using HDBSCAN (min_cluster_size, min_samples)
    - Extracts topic representations using class-based TF-IDF (c-TF-IDF)
    - Optionally reduces topics via hierarchical merging (nr_topics)
    - Saves trained model, topic summaries, and document assignments

    Hyperparameter choices balance topic granularity and coverage for datasets
    with imbalanced category distributions (e.g., 60% street lighting complaints).
    Default settings discover 6-8 natural clusters with ~15-20% outliers.

    Parameters
    ----------
    min_topic_size : int, default=40
        Minimum documents required to form a topic (HDBSCAN min_cluster_size).
        For N ≈ 900 documents, min_topic_size=40 (4.3% of data) allows smaller
        categories to emerge while preventing micro-cluster fragmentation.
    nr_topics : int, str, or None, default=None
        Post-hoc topic reduction strategy:
        - None: Preserve all natural HDBSCAN clusters (recommended).
        - int: Merge to approximately this many topics via hierarchical clustering.
        - "auto": Automatically determine merge threshold based on similarity.
    n_neighbors : int, default=30
        UMAP neighborhood size. Controls local vs. global structure balance.
        Lower values (5-15) preserve local patterns; higher values (50+) emphasize
        global structure. For N ≈ 900 documents, n_neighbors=30 is optimal.
    min_samples : int, default=1
        HDBSCAN outlier sensitivity. Lower values (1-3) reduce outliers; higher
        values (5-10) increase cluster density but exclude more documents.
        For small datasets, min_samples=1 maximizes topic assignment coverage.

    Returns
    -------
    None
        Persists artifacts under BERTOPIC_TOPIC_DIR:
        - bertopic_model/ (serialized BERTopic model)
        - bertopic_topic_info.csv (topic IDs, counts, representative terms)
        - bertopic_doc_topics.csv (per-document topic assignments)  
    """

    df = load_cleaned_for_bertopic(keep_id=True)
    texts = df["bertopic_description"].fillna("").astype(str).tolist()

    embedding_model = SentenceTransformer(
        BERTOPIC_EMBEDDING_MODEL,
        device=BERTOPIC_DEVICE
    )

    # Stopwords are only for topic representation (c-TF-IDF), not embeddings
    nlp = spacy.load("de_core_news_sm")
    stopwords = list(nlp.Defaults.stop_words) + [
        "bitte", "danke", "sehr", "geehrt", "freundlich", "grüße", "hallo",
        "str", "straße", "strasse", "nr"  # common address artifacts
    ]

    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        min_df=2,  # Reduced from 5 to capture more representative terms
        ngram_range=(1, 2),
    )

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=10,  # Sufficient dimensions to separate distinct categories
        min_dist=0.0,     # Allow tighter packing for better clustering
        metric="cosine",
        random_state=BERTOPIC_UMAP_RANDOM_STATE,
    )

    min_cluster_size = int(min_topic_size)

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(texts)

    BERTOPIC_TOPIC_DIR.mkdir(parents=True, exist_ok=True)
    topic_model.save(BERTOPIC_TOPIC_DIR / "bertopic_model")

    info_df = topic_model.get_topic_info()
    info_df.to_csv(BERTOPIC_TOPIC_DIR / "bertopic_topic_info.csv", index=False)

    out_assign = df.copy()
    out_assign["topic_id"] = topics
    out_assign.to_csv(BERTOPIC_TOPIC_DIR / "bertopic_doc_topics.csv", index=False)

    n_outliers = int((np.array(topics) == -1).sum())
    outlier_pct = (n_outliers / len(topics)) * 100 if len(topics) > 0 else 0

    print("BERTopic complete.")
    print(f"Topics discovered (excl. outliers): {info_df.shape[0] - 1}")
    print(f"Outliers (-1 assignments): {n_outliers} / {len(topics)} ({outlier_pct:.1f}%)")
    print(f"Target outlier rate: <15% | Actual: {outlier_pct:.1f}%")
    print(f"Saved model to: {BERTOPIC_TOPIC_DIR / 'bertopic_model'}")
    print(f"Saved topic info to: {BERTOPIC_TOPIC_DIR / 'bertopic_topic_info.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run topic modeling pipelines (LDA and/or BERTopic)"
    )

    parser.add_argument("--lda", action="store_true", help="Run LDA pipelines (BoW and TF-IDF)")
    parser.add_argument("--bertopic", action="store_true", help="Run BERTopic pipeline")

    args = parser.parse_args()

    # default behavior when run from IDE button (no args)
    if not args.lda and not args.bertopic:
        args.lda = True
        args.bertopic = True

    if args.lda:
        run_lda_pipelines(num_topics=5, passes=30, iterations=250)

    if args.bertopic:
        run_bertopic_pipeline(min_topic_size=40, n_neighbors=30, min_samples=1, nr_topics=None)