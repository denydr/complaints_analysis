"""
visualization.py
----------------

Visualization pipeline for topic modeling results from LDA (BoW + TF-IDF) and BERTopic.
Supports both unlabeled (keyword-based) and LLM-labeled (human-readable) topic names
with bilingual output (German and English).

This module implements three primary objectives:
1. Compare BoW vs TF-IDF vectorization techniques
2. Extract and visualize ALL predominant topics from LDA
3. Extract and visualize ALL predominant topics from BERTopic


Outputs
-------
- **Standard visualizations** (saved to `results/`):
    - Files 00-04, 06, 08: Always generated (comparison metrics, topic balance, etc.)
    - Uses keyword-based topic names from model outputs

- **LLM-labeled visualizations** (saved to language-specific subdirectories):
    - `results/german/`: Files 05, 07, 09-14 with German LLM-generated topic names
    - `results/english/`: Files 05, 07, 09-14 with English LLM-generated topic names
    - Uses human-readable topic labels from `topic_labeling.py`


How to Run
----------
1) Basic visualization (no LLM labels):

    python -m src.visualization

2) German LLM-labeled visualizations:

    python -m src.visualization --german

3) Bilingual visualizations (German + English):

    python -m src.visualization --german --english


Prerequisites
-------------
- Must run `topic_modeling.py` first to generate model artifacts
- For LLM-labeled visualizations, must run topic_modeling with `--use-llm` flag
- Language-specific visualizations require corresponding labeled artifacts:
    - `--german` requires artifacts in `topic_models/*/german/`
    - `--english` requires artifacts in `topic_models/*/english/`


File Mapping
------------
- 00-04: Comparison metrics (always in main results/)
- 05: LDA topic-word tables with labels (language-specific)
- 06: LDA predominance charts (main results/)
- 07: BERTopic topic info with labels (language-specific)
- 08: BERTopic predominance chart (main results/)
- 09: pyLDAvis interactive LDA visualization (language-specific)
- 10-14: BERTopic interactive visualizations (language-specific)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# LDA interactive visualization
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import matutils
from gensim.corpora import Dictionary
from scipy.sparse import load_npz

from src.config import (
    RESULTS_DIR,
    RESULTS_DIR_DE,
    RESULTS_DIR_EN,
    RANDOM_STATE,
)
from src.data_loader import (
    load_lda_bow_doc_topics,
    load_lda_bow_info,
    load_lda_bow_k_sweep,
    load_lda_bow_topics,
    load_lda_bow_model,
    load_lda_tfidf_doc_topics,
    load_lda_tfidf_info,
    load_lda_tfidf_k_sweep,
    load_lda_tfidf_topics,
    load_lda_tfidf_model,
    load_bertopic_topic_info,
    load_bertopic_doc_topics,
    load_bertopic_model,
    load_lda_vectorized_artifacts,
    # Labeled topic loaders
    load_lda_labeled_topics,
    load_lda_labeled_doc_topics,
    load_bertopic_labeled_topics,
    load_bertopic_labeled_doc_topics,
    load_bertopic_model_labeled,
)

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")


# =====================================================================
# Helper Functions
# =====================================================================

def _get_results_dir(language: str = None) -> Path:
    """
    Get the appropriate results directory based on language.

    Parameters
    ----------
    language : str, optional
        Language code ('de', 'en', or None for base directory)

    Returns
    -------
    Path
        Results directory for the specified language
    """
    if language == 'de':
        return RESULTS_DIR_DE
    elif language == 'en':
        return RESULTS_DIR_EN
    else:
        return RESULTS_DIR


def _get_file_suffix(language: str = None) -> str:
    """
    Get filename suffix based on language.

    Parameters
    ----------
    language : str, optional
        Language code ('de', 'en', or None)

    Returns
    -------
    str
        Filename suffix ('' for German/None, '_en' for English)
    """
    return '_en' if language == 'en' else ''


# =====================================================================
# OBJECTIVE 1: BoW vs TF-IDF Comparison
# =====================================================================

def compare_vectorization_coherence(save_fig=True):
    """
    Compare coherence scores between BoW and TF-IDF LDA models.

    Parameters
    ----------
    save_fig : bool, default=True
        If True, save figure to RESULTS_DIR.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    bow_info = load_lda_bow_info()
    tfidf_info = load_lda_tfidf_info()

    methods = ['BoW', 'TF-IDF']
    coherences = [bow_info['coherence_c_v'], tfidf_info['coherence_c_v']]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(methods, coherences, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    ax.set_ylabel('Coherence Score (c_v)', fontsize=13, fontweight='bold')
    ax.set_title('LDA Coherence Comparison: BoW vs TF-IDF', fontsize=15, fontweight='bold')
    ax.set_ylim([0, max(coherences) * 1.15])
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, coherences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_fig:
        output_path = RESULTS_DIR / "01_coherence_comparison.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def compare_k_selection(save_fig=True):
    """
    Compare optimal K selection and grid search results.

    Parameters
    ----------
    save_fig : bool, default=True
        If True, save figures to RESULTS_DIR.

    Returns
    -------
    tuple of plt.Figure
        (K comparison bar chart, grid search line plot)
    """
    bow_info = load_lda_bow_info()
    tfidf_info = load_lda_tfidf_info()
    bow_sweep = load_lda_bow_k_sweep()
    tfidf_sweep = load_lda_tfidf_k_sweep()

    # Figure 1: K comparison
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    methods = ['BoW', 'TF-IDF']
    k_values = [bow_info['num_topics'], tfidf_info['num_topics']]
    colors = ['#3498db', '#e74c3c']

    bars = ax1.bar(methods, k_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Topics (K)', fontsize=13, fontweight='bold')
    ax1.set_title('Optimal K Selection: BoW vs TF-IDF', fontsize=15, fontweight='bold')
    ax1.set_ylim([0, max(k_values) * 1.2])
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, k_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'K={val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_fig:
        output_path = RESULTS_DIR / "02_k_selection_comparison.png"
        fig1.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    # Figure 2: Grid search
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(bow_sweep['k'], bow_sweep['coherence_c_v'],
             marker='o', linewidth=2.5, markersize=8, label='BoW', color='#3498db')
    ax2.plot(tfidf_sweep['k'], tfidf_sweep['coherence_c_v'],
             marker='s', linewidth=2.5, markersize=8, label='TF-IDF', color='#e74c3c')

    best_k_bow = bow_sweep.loc[bow_sweep['coherence_c_v'].idxmax(), 'k']
    best_k_tfidf = tfidf_sweep.loc[tfidf_sweep['coherence_c_v'].idxmax(), 'k']

    ax2.axvline(best_k_bow, color='#3498db', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axvline(best_k_tfidf, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=2)

    ax2.set_xlabel('Number of Topics (K)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Coherence Score (c_v)', fontsize=13, fontweight='bold')
    ax2.set_title('LDA Grid Search: Coherence vs K', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_fig:
        output_path = RESULTS_DIR / "03_grid_search_coherence.png"
        fig2.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig1, fig2


def compare_topic_balance(save_fig=True):
    """
    Compare topic balance (document distribution) between BoW and TF-IDF.

    Parameters
    ----------
    save_fig : bool, default=True
        If True, save figure to RESULTS_DIR.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    bow_doc_topics = load_lda_bow_doc_topics()
    tfidf_doc_topics = load_lda_tfidf_doc_topics()

    # Count documents per topic (dominant topic)
    bow_counts_raw = bow_doc_topics['dominant_topic'].value_counts().sort_index()
    tfidf_counts_raw = tfidf_doc_topics['dominant_topic'].value_counts().sort_index()

    # Get actual K from trained models (not from value_counts which misses empty topics)
    bow_info = load_lda_bow_info()
    tfidf_info = load_lda_tfidf_info()

    bow_counts = bow_counts_raw.reindex(range(bow_info['num_topics']), fill_value=0)
    tfidf_counts = tfidf_counts_raw.reindex(range(tfidf_info['num_topics']), fill_value=0)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # BoW
    colors_bow = plt.cm.Blues(np.linspace(0.4, 0.9, len(bow_counts)))
    bars1 = ax1.bar(bow_counts.index, bow_counts.values, color=colors_bow,
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Topic ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Documents', fontsize=12, fontweight='bold')
    ax1.set_title(f'BoW Topic Balance (K={len(bow_counts)})', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, bow_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # TF-IDF
    colors_tfidf = plt.cm.Reds(np.linspace(0.4, 0.9, len(tfidf_counts)))
    bars2 = ax2.bar(tfidf_counts.index, tfidf_counts.values, color=colors_tfidf,
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Topic ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Documents', fontsize=12, fontweight='bold')
    ax2.set_title(f'TF-IDF Topic Balance (K={len(tfidf_counts)})', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, tfidf_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_fig:
        output_path = RESULTS_DIR / "04_topic_balance_comparison.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def create_vectorization_comparison_summary(save_csv=True):
    """
    Create summary table comparing BoW and TF-IDF metrics.

    Parameters
    ----------
    save_csv : bool, default=True
        If True, save table to RESULTS_DIR.

    Returns
    -------
    pd.DataFrame
        Comparison summary table.
    """
    bow_info = load_lda_bow_info()
    tfidf_info = load_lda_tfidf_info()
    bow_doc_topics = load_lda_bow_doc_topics()
    tfidf_doc_topics = load_lda_tfidf_doc_topics()

    # Calculate balance metrics (standard deviation of document counts)
    bow_counts_raw = bow_doc_topics['dominant_topic'].value_counts()
    tfidf_counts_raw = tfidf_doc_topics['dominant_topic'].value_counts()

    # Reindex to include all topics (including empty ones with 0 documents)
    bow_counts = bow_counts_raw.reindex(range(bow_info['num_topics']), fill_value=0)
    tfidf_counts = tfidf_counts_raw.reindex(range(tfidf_info['num_topics']), fill_value=0)

    bow_balance = bow_counts.std() / bow_counts.mean()
    tfidf_balance = tfidf_counts.std() / tfidf_counts.mean()

    summary = pd.DataFrame({
        'Metric': ['Coherence (c_v)', 'Optimal K', 'Total Documents', 'Balance (CV)', 'Interpretation'],
        'BoW': [
            f"{bow_info['coherence_c_v']:.4f}",
            bow_info['num_topics'],
            bow_info['n_docs'],
            f"{bow_balance:.2f}",
            'Better balance'
        ],
        'TF-IDF': [
            f"{tfidf_info['coherence_c_v']:.4f}",
            tfidf_info['num_topics'],
            tfidf_info['n_docs'],
            f"{tfidf_balance:.2f}",
            'Higher imbalance'
        ]
    })

    if save_csv:
        output_path = RESULTS_DIR / "00_vectorization_comparison_summary.csv"
        summary.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")

    return summary


# =====================================================================
# OBJECTIVE 2: LDA Predominant Topics (ALL Topics)
# =====================================================================

def extract_lda_all_topics(model_type='bow', top_n_words=10, save_csv=True, language=None):
    """
    Extract ALL topics from LDA model with predominance information.

    Parameters
    ----------
    model_type : str, default='bow'
        'bow' or 'tfidf'
    top_n_words : int, default=10
        Number of top words to extract per topic
    save_csv : bool, default=True
        If True, save table to results directory.
    language : str, optional
        Language for labeled topics ('de', 'en', or None for unlabeled)

    Returns
    -------
    pd.DataFrame
        Table with all topics, document counts, and top words.
    """
    # Try to load labeled data if language is specified
    if language in ['de', 'en']:
        try:
            topics_df = load_lda_labeled_topics(model_type=model_type, language=language)
            doc_topics = load_lda_labeled_doc_topics(model_type=model_type, language=language)
            has_labels = True
        except FileNotFoundError:
            print(f"⚠️  Labeled data not found for language '{language}'. Falling back to unlabeled data.")
            language = None
            has_labels = False
    else:
        has_labels = False

    # Load unlabeled data if no language specified or labeled data not found
    if not has_labels:
        if model_type == 'bow':
            doc_topics = load_lda_bow_doc_topics()
            topics_df = load_lda_bow_topics()
        else:
            doc_topics = load_lda_tfidf_doc_topics()
            topics_df = load_lda_tfidf_topics()

    # Count documents per topic
    topic_counts = doc_topics['dominant_topic'].value_counts().sort_index()

    # Extract top words per topic
    all_topics = []
    for topic_id in sorted(topics_df['topic_id'].unique()):
        topic_data = topics_df[topics_df['topic_id'] == topic_id]
        top_words = topic_data.nlargest(top_n_words, 'weight')['word'].tolist()

        doc_count = topic_counts.get(topic_id, 0)
        pct = (doc_count / len(doc_topics)) * 100

        row = {
            'Topic_ID': topic_id,
            'Doc_Count': doc_count,
            'Percentage': f"{pct:.1f}%",
            'Top_Words': ', '.join(top_words),
            'Is_Predominant': doc_count == topic_counts.max()
        }

        # Add topic name if labels available
        if has_labels and 'topic_name' in topic_data.columns:
            topic_name = topic_data['topic_name'].iloc[0]
            row = {'Topic_ID': topic_id, 'Topic_Name': topic_name, **{k: v for k, v in row.items() if k != 'Topic_ID'}}

        all_topics.append(row)

    df_topics = pd.DataFrame(all_topics)

    if save_csv:
        results_dir = _get_results_dir(language)
        results_dir.mkdir(parents=True, exist_ok=True)
        suffix = _get_file_suffix(language)
        output_path = results_dir / f"05_lda_{model_type}_all_topics{suffix}.csv"
        df_topics.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")

    return df_topics


def visualize_lda_all_topics_predominance(model_type='bow', save_fig=True):
    """
    Visualize document predominance for ALL LDA topics.

    Parameters
    ----------
    model_type : str, default='bow'
        'bow' or 'tfidf'
    save_fig : bool, default=True
        If True, save figure to RESULTS_DIR.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    if model_type == 'bow':
        doc_topics = load_lda_bow_doc_topics()
        prefix = 'BoW'
        color_palette = 'Blues'
    else:
        doc_topics = load_lda_tfidf_doc_topics()
        prefix = 'TF-IDF'
        color_palette = 'Reds'

    topic_counts = doc_topics['dominant_topic'].value_counts().sort_index()
    max_count = topic_counts.max()

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = []
    for count in topic_counts.values:
        if count == max_count:
            colors.append('#2ecc71')  # Highlight predominant in green
        else:
            colors.append(plt.colormaps[color_palette](0.6))

    bars = ax.bar(topic_counts.index, topic_counts.values, color=colors,
                  edgecolor='black', linewidth=2, alpha=0.8)

    ax.set_xlabel('Topic ID', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Documents', fontsize=13, fontweight='bold')
    ax.set_title(f'LDA {prefix}: Document Predominance Across ALL Topics (K={len(topic_counts)})',
                 fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for i, (topic_id, count) in enumerate(zip(topic_counts.index, topic_counts.values)):
        pct = (count / topic_counts.sum()) * 100
        label = f'{count}\n({pct:.1f}%)'
        if count == max_count:
            label += '\n★'  # Star for predominant
        ax.text(i, count + max_count * 0.02, label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_fig:
        output_path = RESULTS_DIR / f"06_lda_{model_type}_predominance.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def visualize_lda_pyldavis(model_type='bow', save_html=True, language=None):
    """
    Create interactive pyLDAvis visualization for LDA model.

    Parameters
    ----------
    model_type : str, default='bow'
        'bow' or 'tfidf'
    save_html : bool, default=True
        If True, save interactive HTML to results directory.
    language : str, optional
        Language for directory structure and keyword translation ('de', 'en', or None)
        If 'en', keywords will be translated to English using labeled data

    Returns
    -------
    pyLDAvis.PreparedData
        Prepared visualization data.
    """
    if model_type == 'bow':
        lda_model = load_lda_bow_model()
        prefix = 'bow'
    else:
        lda_model = load_lda_tfidf_model()
        prefix = 'tfidf'

    # Load vectorized data
    artifacts = load_lda_vectorized_artifacts()
    X_sparse = artifacts['X_bow'] if model_type == 'bow' else artifacts['X_tfidf']

    # Convert to Gensim corpus format
    corpus = matutils.Sparse2Corpus(X_sparse, documents_columns=False)

    # Check if id2word is a proper Dictionary object or a plain dict
    if hasattr(lda_model.id2word, 'token2id'):
        # Already a proper Dictionary object with all attributes
        dictionary = lda_model.id2word
    else:
        # id2word is a plain dict - need to create a proper Dictionary
        # pyLDAvis requires a Dictionary with num_docs, num_pos, dfs, cfs attributes
        dictionary = Dictionary()
        dictionary.id2token = dict(lda_model.id2word)
        dictionary.token2id = {word: id_ for id_, word in dictionary.id2token.items()}

        # Compute required attributes from the corpus
        dictionary.num_docs = 0
        dictionary.num_pos = 0
        dictionary.num_nnz = 0
        dictionary.dfs = {}  # document frequency: how many docs each word appears in
        dictionary.cfs = {}  # collection frequency: total count of each word

        # Iterate through corpus to compute frequencies
        for doc in corpus:
            dictionary.num_docs += 1
            doc_words = set()
            for word_id, count in doc:
                dictionary.num_pos += count
                dictionary.num_nnz += 1
                dictionary.cfs[word_id] = dictionary.cfs.get(word_id, 0) + count
                doc_words.add(word_id)
            # Update document frequencies
            for word_id in doc_words:
                dictionary.dfs[word_id] = dictionary.dfs.get(word_id, 0) + 1

        print(f"  ✓ Constructed Dictionary with {len(dictionary)} words from corpus")

    # Create visualization with t-SNE and random_state for reproducibility
    vis = gensimvis.prepare(
        lda_model,
        corpus,
        dictionary,
        mds='tsne',
        sort_topics=False,
        R=10
    )

    if save_html:
        results_dir = _get_results_dir(language)
        results_dir.mkdir(parents=True, exist_ok=True)
        suffix = _get_file_suffix(language)
        output_path = results_dir / f"09_lda_{prefix}_pyldavis{suffix}.html"
        pyLDAvis.save_html(vis, str(output_path))
        print(f"✓ Saved: {output_path}")

    return vis


# =====================================================================
# OBJECTIVE 3: BERTopic Predominant Topics (ALL Topics)
# =====================================================================

def extract_bertopic_all_topics(top_n_words=10, save_csv=True, language=None):
    """
    Extract ALL topics from BERTopic model with predominance information.

    Parameters
    ----------
    top_n_words : int, default=10
        Number of top words to show per topic
    save_csv : bool, default=True
        If True, save table to results directory.
    language : str, optional
        Language for labeled topics ('de', 'en', or None for unlabeled)

    Returns
    -------
    pd.DataFrame
        Table with all topics, document counts, and top words.
    """
    # Try to load labeled data if language is specified
    if language in ['de', 'en']:
        try:
            topic_info = load_bertopic_labeled_topics(language=language)
            doc_topics = load_bertopic_labeled_doc_topics(language=language)
            has_labels = True
        except FileNotFoundError:
            print(f"⚠️  Labeled data not found for language '{language}'. Falling back to unlabeled data.")
            topic_info = load_bertopic_topic_info()
            doc_topics = load_bertopic_doc_topics()
            has_labels = False
    else:
        topic_info = load_bertopic_topic_info()
        doc_topics = load_bertopic_doc_topics()
        has_labels = False

    total_docs = len(doc_topics)
    max_count = topic_info[topic_info['Topic'] != -1]['Count'].max()

    all_topics = []
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        count = row['Count']
        pct = (count / total_docs) * 100

        # Parse top words from representation
        try:
            top_words = eval(row['Representation'])[:top_n_words]
            top_words_str = ', '.join(top_words)
        except:
            top_words_str = str(row['Representation'])[:100]

        # Use LLM-generated name if available, otherwise use default
        if has_labels and 'Name' in row:
            topic_name = row['Name']
        else:
            topic_name = row.get('Name', f"Topic {topic_id}")

        all_topics.append({
            'Topic_ID': topic_id,
            'Topic_Name': topic_name,
            'Doc_Count': count,
            'Percentage': f"{pct:.1f}%",
            'Top_Words': top_words_str,
            'Is_Predominant': (count == max_count and topic_id != -1),
            'Is_Outlier': topic_id == -1
        })

    df_topics = pd.DataFrame(all_topics)

    if save_csv:
        results_dir = _get_results_dir(language)
        results_dir.mkdir(parents=True, exist_ok=True)
        suffix = _get_file_suffix(language)
        output_path = results_dir / f"07_bertopic_all_topics{suffix}.csv"
        df_topics.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")

    return df_topics


def visualize_bertopic_all_topics_predominance(save_fig=True):
    """
    Visualize document predominance for ALL BERTopic topics.

    Parameters
    ----------
    save_fig : bool, default=True
        If True, save figure to RESULTS_DIR.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    topic_info = load_bertopic_topic_info()

    # Separate outliers from regular topics
    regular_topics = topic_info[topic_info['Topic'] != -1].copy()
    outliers = topic_info[topic_info['Topic'] == -1].copy()

    max_count = regular_topics['Count'].max()
    total_docs = topic_info['Count'].sum()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data
    all_data = pd.concat([regular_topics, outliers])
    all_data = all_data.sort_values('Topic')

    colors = []
    for _, row in all_data.iterrows():
        if row['Topic'] == -1:
            colors.append('#95a5a6')  # Gray for outliers
        elif row['Count'] == max_count:
            colors.append('#2ecc71')  # Green for predominant
        else:
            colors.append('#3498db')  # Blue for regular

    bars = ax.bar(range(len(all_data)), all_data['Count'].values,
                  color=colors, edgecolor='black', linewidth=2, alpha=0.8)

    ax.set_xlabel('Topic', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Documents', fontsize=13, fontweight='bold')
    ax.set_title(f'BERTopic: Document Predominance Across ALL Topics (Total={len(all_data)} topics)',
                 fontsize=15, fontweight='bold')
    ax.set_xticks(range(len(all_data)))
    ax.set_xticklabels([f"T{int(t)}" if t != -1 else "OUT" for t in all_data['Topic'].values],
                       rotation=0, fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for i, (idx, row) in enumerate(all_data.iterrows()):
        count = row['Count']
        pct = (count / total_docs) * 100
        label = f'{count}\n({pct:.1f}%)'

        if row['Topic'] != -1 and count == max_count:
            label += '\n★'  # Star for predominant

        ax.text(i, count + max_count * 0.02, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Predominant Topic'),
        Patch(facecolor='#3498db', edgecolor='black', label='Regular Topics'),
        Patch(facecolor='#95a5a6', edgecolor='black', label='Outliers')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()

    if save_fig:
        output_path = RESULTS_DIR / "08_bertopic_predominance.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def visualize_bertopic_interactive_all(save_html=True, language=None):
    """
    Create all BERTopic built-in interactive visualizations.

    Parameters
    ----------
    save_html : bool, default=True
        If True, save interactive HTML files to results directory.
    language : str, optional
        Language for labeled model ('de', 'en', or None for unlabeled)

    Returns
    -------
    dict
        Dictionary of Plotly figure objects.
    """
    # Try to load labeled model if language is specified
    if language in ['de', 'en']:
        try:
            model = load_bertopic_model_labeled(language=language)
            topic_info = load_bertopic_labeled_topics(language=language)
            doc_topics = load_bertopic_labeled_doc_topics(language=language)
        except FileNotFoundError:
            print(f"⚠️  Labeled model not found for language '{language}'. Falling back to unlabeled model.")
            model = load_bertopic_model()
            topic_info = load_bertopic_topic_info()
            doc_topics = load_bertopic_doc_topics()
            language = None
    else:
        model = load_bertopic_model()
        topic_info = load_bertopic_topic_info()
        doc_topics = load_bertopic_doc_topics()

    # Load original documents for some visualizations
    from src.data_loader import load_cleaned_for_bertopic
    df_bertopic = load_cleaned_for_bertopic()
    docs = df_bertopic['bertopic_description'].tolist()

    # Get output directory and suffix
    results_dir = _get_results_dir(language)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = _get_file_suffix(language)

    figures = {}

    # 1. Topic space visualization (2D UMAP projection)
    try:
        fig1 = model.visualize_topics(width=1200, height=800, custom_labels=True)
        figures['topic_space'] = fig1
        if save_html:
            output_path = results_dir / f"10_bertopic_topic_space{suffix}.html"
            fig1.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create topic space visualization: {e}")

    # 2. Hierarchical topic clustering
    try:
        fig2 = model.visualize_hierarchy(width=1200, height=800, custom_labels=True)
        figures['hierarchy'] = fig2
        if save_html:
            output_path = results_dir / f"11_bertopic_hierarchy{suffix}.html"
            fig2.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create hierarchy visualization: {e}")

    # 3. Top words per topic (bar chart)
    try:
        num_topics = len(topic_info[topic_info['Topic'] != -1])
        fig3 = model.visualize_barchart(top_n_topics=num_topics, n_words=10, width=1000, height=600, custom_labels=True)
        figures['barchart'] = fig3
        if save_html:
            output_path = results_dir / f"12_bertopic_barchart{suffix}.html"
            fig3.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create barchart visualization: {e}")

    # 4. Topic similarity heatmap
    try:
        fig4 = model.visualize_heatmap(custom_labels=True)
        figures['heatmap'] = fig4
        if save_html:
            output_path = results_dir / f"13_bertopic_heatmap{suffix}.html"
            fig4.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create heatmap visualization: {e}")

    # 5. Document visualization (if documents are available)
    try:
        fig5 = model.visualize_documents(docs, reduced_embeddings=model.umap_model.transform(model.embedding_model.embed(docs)), custom_labels=True)
        figures['documents'] = fig5
        if save_html:
            output_path = results_dir / f"14_bertopic_documents{suffix}.html"
            fig5.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create documents visualization: {e}")

    return figures


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_visualization_pipeline(use_llm_labels=False, generate_english=False):
    """
    Run complete visualization pipeline for all 3 primary objectives.

    Generates:
    - BoW vs TF-IDF comparison visualizations (always in main results/)
    - LDA predominant topics analysis (ALL topics)
    - BERTopic predominant topics analysis (ALL topics)
    - Optional: Bilingual labeled visualizations (german/ and english/ subdirs)

    Parameters
    ----------
    use_llm_labels : bool, default=False
        If True, generate visualizations with LLM-generated topic labels
        and save to language-specific directories (results/german/)
    generate_english : bool, default=False
        If True, also generate English versions (requires use_llm_labels=True)
        Saves to results/english/

    Returns
    -------
    None
        All outputs saved to RESULTS_DIR (and language subdirs if applicable)
    """
    print("=" * 80)
    print("TOPIC MODELING VISUALIZATION PIPELINE")
    print("=" * 80)

    # Objective 1: BoW vs TF-IDF Comparison (always in main results/, no language subdirs)
    print("\n[1/3] Comparing BoW vs TF-IDF Vectorization...")
    print("-" * 80)
    create_vectorization_comparison_summary()
    compare_vectorization_coherence()
    compare_k_selection()
    compare_topic_balance()

    # Determine which languages to generate
    languages = []
    if use_llm_labels:
        languages.append('de')
        if generate_english:
            languages.append('en')
    else:
        languages.append(None)  # Unlabeled, main results/

    # Generate visualizations for each language
    for lang in languages:
        lang_label = f" ({lang.upper()})" if lang else ""

        # Objective 2: LDA Predominant Topics
        print(f"\n[2/3] Extracting ALL LDA Predominant Topics{lang_label}...")
        print("-" * 80)
        extract_lda_all_topics(model_type='bow', language=lang)
        extract_lda_all_topics(model_type='tfidf', language=lang)

        # Objective 3: BERTopic Predominant Topics
        print(f"\n[3/3] Extracting ALL BERTopic Predominant Topics{lang_label}...")
        print("-" * 80)
        extract_bertopic_all_topics(language=lang)

        # Interactive Visualizations
        print(f"\n[BONUS] Generating Interactive Visualizations{lang_label}...")
        print("-" * 80)
        if lang != 'en':
            print(f"pyLDAvis (BoW){lang_label}...")
            visualize_lda_pyldavis(model_type='bow', language=lang)
            print(f"pyLDAvis (TF-IDF){lang_label}...")
            visualize_lda_pyldavis(model_type='tfidf', language=lang)
        print(f"BERTopic interactive plots{lang_label}...")
        visualize_bertopic_interactive_all(language=lang)

    # Static visualizations (only when generating unlabeled - files 06, 08)
    if None in languages:
        print("\n[STATIC] Generating Static Visualizations...")
        print("-" * 80)
        visualize_lda_all_topics_predominance(model_type='bow')
        visualize_lda_all_topics_predominance(model_type='tfidf')
        visualize_bertopic_all_topics_predominance()

    print("\n" + "=" * 80)
    results_msg = f"✓ COMPLETE! Visualizations saved to: {RESULTS_DIR}"
    if use_llm_labels:
        results_msg += f"\n  - German: {RESULTS_DIR_DE}"
        if generate_english:
            results_msg += f"\n  - English: {RESULTS_DIR_EN}"
    print(results_msg)
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate topic modeling visualizations")
    parser.add_argument("--german", action="store_true", help="Generate German labeled visualizations")
    parser.add_argument("--english", action="store_true", help="Generate English labeled visualizations")
    args = parser.parse_args()

    use_llm = args.german or args.english
    run_visualization_pipeline(use_llm_labels=use_llm, generate_english=args.english)