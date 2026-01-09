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

    python -m src.topic_modeling --lda

This will:
- Train LDA on BoW
- Train LDA on TF-IDF
- Save LDA outputs
- Skip BERTopic entirely

3) Run ONLY BERTopic

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
from collections import Counter

from gensim import matutils
from gensim.corpora import Dictionary
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
        - doc_id
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

def _select_best_k_by_coherence(
    X_sparse,
    vocab: list[str],
    texts_tokens: list[list[str]],
    k_min: int,
    k_max: int,
    passes: int,
    iterations: int,
    random_state: int,
    out_csv: Path,
    out_prefix: str,
):
    """
    Sweep K (num_topics) and select the best model by c_v coherence.

    This function is used to automatically choose the number of topics for LDA:
    - trains temporary LDA models for each K
    - computes c_v coherence
    - saves a sweep table (CSV)
    - returns the best K (highest coherence)

    Parameters
    ----------
    X_sparse : scipy.sparse matrix
        Document-term matrix (BoW or TF-IDF).
    vocab : list[str]
        Vocabulary aligned with columns of X_sparse.
    texts_tokens : list[list[str]]
        Tokenized documents (for coherence), aligned with X_sparse rows.
    k_min : int
        Minimum number of topics to try.
    k_max : int
        Maximum number of topics to try (inclusive).
    passes : int
        Passes for LDA training during sweep.
    iterations : int
        Iterations for LDA training during sweep.
    random_state : int
        Random seed for reproducibility.
    out_csv : Path
        Output CSV path for sweep results.
    out_prefix : str
        Prefix used only for logging clarity.

    Returns
    -------
    tuple[int, pd.DataFrame]
        (best_k, results_df)
    """

    if int(k_max) < int(k_min):
        raise ValueError("k_max must be >= k_min")

    corpus = matutils.Sparse2Corpus(X_sparse, documents_columns=False)
    id2word = dict(enumerate(vocab))

    vocab_set = set(vocab)
    texts_tokens_aligned = [[tok for tok in doc if tok in vocab_set] for doc in texts_tokens]
    dictionary = Dictionary(texts_tokens_aligned)

    rows = []
    best_k = None
    best_coh = -1.0

    for k in range(int(k_min), int(k_max) + 1):
        lda_tmp = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=int(k),
            passes=int(passes),
            iterations=int(iterations),
            random_state=int(random_state),
            eval_every=None,
            alpha="auto",
            eta="auto",
            chunksize=100,
        )

        coherence_model = CoherenceModel(
            model=lda_tmp,
            texts=texts_tokens_aligned,
            dictionary=dictionary,
            coherence="c_v",
        )
        coh = float(coherence_model.get_coherence())

        doc_topics = [lda_tmp.get_document_topics(doc, minimum_probability=0.0) for doc in corpus]
        dominant_topics = [max(topics, key=lambda x: x[1])[0] if topics else -1 for topics in doc_topics]
        topic_counts = Counter(dominant_topics)
        empty_topics = sum(1 for topic_id in range(k) if topic_counts.get(topic_id, 0) == 0)

        rows.append(
            {
                "out_prefix": out_prefix,
                "k": int(k),
                "coherence_c_v": coh,
                "empty_topics": empty_topics,
                "passes": int(passes),
                "iterations": int(iterations),
                "random_state": int(random_state),
            }
        )

        validation_status = "valid" if empty_topics == 0 else f"invalid ({empty_topics} empty)"
        print(f"  K={k}: coherence={coh:.4f}, {validation_status}")

        # Select best K by coherence (only among valid models with no empty topics)
        if empty_topics == 0 and coh > best_coh:
            best_coh = coh
            best_k = int(k)

    results_df = pd.DataFrame(rows)

    if best_k is None:
        print(f"⚠ Warning: No valid models found for {out_prefix} (all K values have empty topics)")
        print(f"   Falling back to K with fewest empty topics, then highest coherence")

        # Find minimum number of empty topics
        min_empty = results_df['empty_topics'].min()
        candidates = results_df[results_df['empty_topics'] == min_empty]

        # Among candidates with min empty topics, select highest coherence
        best_idx = candidates['coherence_c_v'].idxmax()
        best_k = int(results_df.loc[best_idx, 'k'])
        best_coh = results_df.loc[best_idx, 'coherence_c_v']

        print(f"   Selected K={best_k} (min_empty={min_empty}, coherence={best_coh:.4f})")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)

    print(f"✅ {out_prefix}: best_k={best_k} (c_v={best_coh:.4f})")
    print(f"   saved sweep results: {out_csv}")

    return best_k, results_df

# -------------------------
# Main pipelines
# -------------------------
def run_lda_pipelines(
    num_topics: int = 5,
    passes: int = 30,
    iterations: int = 250,
    auto_k: bool = True,
    k_min: int = 3,
    k_max: int = 10,
) -> None:
    """
    Train and compare two LDA models:
    1) LDA trained on BoW matrix
    2) LDA trained on TF-IDF matrix (comparison run)

    By default, the pipeline automatically selects the number of topics (K)
    separately for BoW and TF-IDF by sweeping a K range and choosing the
    highest c_v coherence.

    Parameters
    ----------
    num_topics : int, default=5
        Fallback number of topics (used only when auto_k=False).
    passes : int, default=30
        Number of passes through the corpus.
    iterations : int, default=250
        Maximum number of iterations per pass.
    auto_k : bool, default=True
        If True, sweep K in [k_min, k_max] and select the best K by c_v coherence
        separately for BoW and TF-IDF.
        If False, train both models with num_topics.
    k_min : int, default=3
        Minimum number of topics (inclusive) to try when auto_k=True.
    k_max : int, default=10
        Maximum number of topics (inclusive) to try when auto_k=True.

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

    # -------------------------
    # Auto-select K (topics)
    # -------------------------
    if auto_k:
        best_k_bow, _ = _select_best_k_by_coherence(
            X_sparse=X_bow,
            vocab=bow_vocab,
            texts_tokens=texts_tokens,
            k_min=k_min,
            k_max=k_max,
            passes=passes,
            iterations=iterations,
            random_state=LDA_RANDOM_STATE,
            out_csv=LDA_TOPIC_DIR / "lda_bow_k_sweep.csv",
            out_prefix="lda_bow",
        )

        best_k_tfidf, _ = _select_best_k_by_coherence(
            X_sparse=X_tfidf,
            vocab=tfidf_vocab,
            texts_tokens=texts_tokens,
            k_min=k_min,
            k_max=k_max,
            passes=passes,
            iterations=iterations,
            random_state=LDA_RANDOM_STATE,
            out_csv=LDA_TOPIC_DIR / "lda_tfidf_k_sweep.csv",
            out_prefix="lda_tfidf",
        )
    else:
        best_k_bow = int(num_topics)
        best_k_tfidf = int(num_topics)

    print(f"Selected K: BoW={best_k_bow}, TF-IDF={best_k_tfidf}")

    # -------------------------
    # Train final models (save artifacts)
    # -------------------------
    _, coh_bow = _train_gensim_lda(
        X_sparse=X_bow,
        vocab=bow_vocab,
        texts_tokens=texts_tokens,
        num_topics=best_k_bow,
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
        num_topics=best_k_tfidf,
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
    reduction, and HDBSCAN density-based clustering.

    The function:
    - Generates semantic document embeddings using SentenceTransformers
    - Reduces embedding dimensionality via UMAP
    - Clusters documents using HDBSCAN
    - Derives topic representations using class-based TF-IDF (c-TF-IDF)
    - Optionally reassigns outlier documents post hoc based on embedding similarity (cosine similarity threshold = 0.6)
    - Optionally merges semantically similar topics after initial discovery via the `nr_topics` parameter
    - Saves the trained model, topic summaries, and per-document topic assignments

  Parameters
    ----------
    min_topic_size : int, default=40
        Minimum number of documents required to form a topic
        (HDBSCAN min_cluster_size). This parameter controls the
        minimum semantic mass needed for a cluster to be retained
        as a topic.
    nr_topics : int, str, or None, default=None
        Optional post-hoc topic reduction setting. If None, all
        clusters discovered by HDBSCAN are retained. If set to an
        integer or "auto", semantically similar topics may be
        merged after initial topic discovery.
    n_neighbors : int, default=30
        Number of neighboring points considered during UMAP
        dimensionality reduction. This parameter influences the
        balance between local and global structure in the embedding
        space.
    min_samples : int, default=1
        HDBSCAN parameter controlling how strictly documents are
        classified as noise. Lower values assign more documents
        to topics, while higher values increase outlier sensitivity.

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
        # Greetings
        "bitte", "danke", "hallo",

        # Closing formulas (base + inflected forms + spelling variants)
        "freundlich", "freundlichen",
        "grüße", "grüßen", "gruß", "gruss", "grussen", "grüssen", "gruesse",
        "herzlich", "herzlichen",
        "hochachtungsvoll",

        # Formal salutations
        "sehr", "geehrt", "geehrte", "geehrter", "geehrten",

        # Common fillers & abbreviations
        "etc", "usw",  # und so weiter
        "mfg", "vg",   # mit freundlichen grüßen, viele grüße

        # Address artifacts
        "str", "straße", "strasse", "nr",
    ]

    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        min_df=2,   # Minimum document frequency for c-TF-IDF topic representations
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
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(texts)

    # Count outliers before reduction
    n_outliers_before = int((np.array(topics) == -1).sum())
    outlier_pct_before = (n_outliers_before / len(topics)) * 100 if len(topics) > 0 else 0

    # Reduce outliers using threshold-based embedding similarity
    # Only reassign outliers with cosine similarity > 0.6 to nearest topic
    # This preserves original c-TF-IDF representations while selectively reassigning
    print(f"\nReducing outliers (initial: {n_outliers_before} / {len(topics)} = {outlier_pct_before:.1f}%)...")
    print("Using embeddings strategy with threshold=0.6 (only reassign confident matches)...")
    new_topics = topic_model.reduce_outliers(texts, topics, strategy="embeddings", threshold=0.6)

    # Count outliers after reduction
    n_outliers_after = int((np.array(new_topics) == -1).sum())
    outlier_pct_after = (n_outliers_after / len(new_topics)) * 100 if len(new_topics) > 0 else 0

    # Note: We intentionally do NOT call update_topics(texts, topics=new_topics) here
    # because that would recalculate c-TF-IDF and degrade topic quality.
    # Instead, llm_topic_labeling.py loads corrected counts from the CSV we save below.

    BERTOPIC_TOPIC_DIR.mkdir(parents=True, exist_ok=True)
    topic_model.save(BERTOPIC_TOPIC_DIR / "bertopic_model")

    # Get topic info and update counts to reflect actual assignments after outlier reduction
    info_df = topic_model.get_topic_info()

    # Recalculate actual document counts from new_topics (after outlier reduction)
    actual_counts = Counter(new_topics)
    info_df['Count'] = info_df['Topic'].apply(lambda t: actual_counts.get(t, 0))

    info_df.to_csv(BERTOPIC_TOPIC_DIR / "bertopic_topic_info.csv", index=False)

    out_assign = df.copy()
    out_assign["topic_id"] = new_topics
    out_assign.to_csv(BERTOPIC_TOPIC_DIR / "bertopic_doc_topics.csv", index=False)

    print("\nBERTopic complete.")
    print(f"Topics discovered (excl. outliers): {info_df.shape[0] - 1}")
    print(f"Outliers before reduction: {n_outliers_before} / {len(topics)} ({outlier_pct_before:.1f}%)")
    print(f"Outliers after reduction:  {n_outliers_after} / {len(new_topics)} ({outlier_pct_after:.1f}%)")
    reduction_count = n_outliers_before - n_outliers_after
    reduction_pct = outlier_pct_before - outlier_pct_after
    print(f"Outliers reduced by: {reduction_count} documents ({reduction_pct:.1f} percentage points)")
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
        run_lda_pipelines(passes=30, iterations=250, auto_k=True)

    if args.bertopic:
        run_bertopic_pipeline(min_topic_size=40, n_neighbors=30, min_samples=1, nr_topics=None)