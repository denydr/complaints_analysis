"""
visualization.py
----------------

Visualization pipeline for topic modeling results from LDA (BoW + TF-IDF) and BERTopic.

This module implements three primary objectives:
1. Compare BoW vs TF-IDF vectorization techniques
2. Extract and visualize ALL predominant topics from LDA
3. Extract and visualize ALL predominant topics from BERTopic

All outputs are saved to the results/ directory.
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

from src.config import RESULTS_DIR, RANDOM_STATE
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
)

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")


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
    bow_counts = bow_doc_topics['dominant_topic'].value_counts().sort_index()
    tfidf_counts = tfidf_doc_topics['dominant_topic'].value_counts().sort_index()

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
    bow_counts = bow_doc_topics['dominant_topic'].value_counts()
    tfidf_counts = tfidf_doc_topics['dominant_topic'].value_counts()

    bow_balance = bow_counts.std() / bow_counts.mean()  # Coefficient of variation
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

def extract_lda_all_topics(model_type='bow', top_n_words=10, save_csv=True):
    """
    Extract ALL topics from LDA model with predominance information.

    Parameters
    ----------
    model_type : str, default='bow'
        'bow' or 'tfidf'
    top_n_words : int, default=10
        Number of top words to extract per topic
    save_csv : bool, default=True
        If True, save table to RESULTS_DIR.

    Returns
    -------
    pd.DataFrame
        Table with all topics, document counts, and top words.
    """
    if model_type == 'bow':
        doc_topics = load_lda_bow_doc_topics()
        topics_df = load_lda_bow_topics()
        prefix = 'BoW'
    else:
        doc_topics = load_lda_tfidf_doc_topics()
        topics_df = load_lda_tfidf_topics()
        prefix = 'TF-IDF'

    # Count documents per topic
    topic_counts = doc_topics['dominant_topic'].value_counts().sort_index()

    # Extract top words per topic
    all_topics = []
    for topic_id in sorted(topics_df['topic_id'].unique()):
        top_words = (topics_df[topics_df['topic_id'] == topic_id]
                     .nlargest(top_n_words, 'weight')['word'].tolist())

        doc_count = topic_counts.get(topic_id, 0)
        pct = (doc_count / len(doc_topics)) * 100

        all_topics.append({
            'Topic_ID': topic_id,
            'Doc_Count': doc_count,
            'Percentage': f"{pct:.1f}%",
            'Top_Words': ', '.join(top_words),
            'Is_Predominant': doc_count == topic_counts.max()
        })

    df_topics = pd.DataFrame(all_topics)

    if save_csv:
        output_path = RESULTS_DIR / f"05_lda_{model_type}_all_topics.csv"
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
            colors.append(plt.cm.get_cmap(color_palette)(0.6))

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


def visualize_lda_pyldavis(model_type='bow', save_html=True):
    """
    Create interactive pyLDAvis visualization for LDA model.

    Parameters
    ----------
    model_type : str, default='bow'
        'bow' or 'tfidf'
    save_html : bool, default=True
        If True, save interactive HTML to RESULTS_DIR.

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

    # Create proper gensim Dictionary object from model's id2word
    # The model's id2word is a plain dict {id: word}, we need a Dictionary object
    id2word_dict = lda_model.id2word
    dictionary = Dictionary()
    dictionary.token2id = {word: idx for idx, word in id2word_dict.items()}
    dictionary.id2token = id2word_dict

    # Create visualization with t-SNE and random_state for reproducibility
    vis = gensimvis.prepare(
        lda_model,
        corpus,
        dictionary,
        mds='tsne',
        sort_topics=False,
        R=42  # Random state for t-SNE projection
    )

    if save_html:
        output_path = RESULTS_DIR / f"09_lda_{prefix}_pyldavis.html"
        pyLDAvis.save_html(vis, str(output_path))
        print(f"✓ Saved: {output_path}")

    return vis


# =====================================================================
# OBJECTIVE 3: BERTopic Predominant Topics (ALL Topics)
# =====================================================================

def extract_bertopic_all_topics(top_n_words=10, save_csv=True):
    """
    Extract ALL topics from BERTopic model with predominance information.

    Parameters
    ----------
    top_n_words : int, default=10
        Number of top words to show per topic
    save_csv : bool, default=True
        If True, save table to RESULTS_DIR.

    Returns
    -------
    pd.DataFrame
        Table with all topics, document counts, and top words.
    """
    topic_info = load_bertopic_topic_info()
    doc_topics = load_bertopic_doc_topics()

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

        all_topics.append({
            'Topic_ID': topic_id,
            'Topic_Name': row['Name'] if 'Name' in row else f"Topic {topic_id}",
            'Doc_Count': count,
            'Percentage': f"{pct:.1f}%",
            'Top_Words': top_words_str,
            'Is_Predominant': (count == max_count and topic_id != -1),
            'Is_Outlier': topic_id == -1
        })

    df_topics = pd.DataFrame(all_topics)

    if save_csv:
        output_path = RESULTS_DIR / "07_bertopic_all_topics.csv"
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


def visualize_bertopic_interactive_all(save_html=True):
    """
    Create all BERTopic built-in interactive visualizations.

    Parameters
    ----------
    save_html : bool, default=True
        If True, save interactive HTML files to RESULTS_DIR.

    Returns
    -------
    dict
        Dictionary of Plotly figure objects.
    """
    model = load_bertopic_model()
    topic_info = load_bertopic_topic_info()
    doc_topics = load_bertopic_doc_topics()

    # Load original documents for some visualizations
    from src.data_loader import load_cleaned_for_bertopic
    df_bertopic = load_cleaned_for_bertopic()
    docs = df_bertopic['bertopic_description'].tolist()

    figures = {}

    # 1. Topic space visualization (2D UMAP projection)
    try:
        fig1 = model.visualize_topics()
        figures['topic_space'] = fig1
        if save_html:
            output_path = RESULTS_DIR / "10_bertopic_topic_space.html"
            fig1.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create topic space visualization: {e}")

    # 2. Hierarchical topic clustering
    try:
        fig2 = model.visualize_hierarchy()
        figures['hierarchy'] = fig2
        if save_html:
            output_path = RESULTS_DIR / "11_bertopic_hierarchy.html"
            fig2.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create hierarchy visualization: {e}")

    # 3. Top words per topic (bar chart)
    try:
        num_topics = len(topic_info[topic_info['Topic'] != -1])
        fig3 = model.visualize_barchart(top_n_topics=num_topics, n_words=10)
        figures['barchart'] = fig3
        if save_html:
            output_path = RESULTS_DIR / "12_bertopic_barchart.html"
            fig3.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create barchart visualization: {e}")

    # 4. Topic similarity heatmap
    try:
        fig4 = model.visualize_heatmap()
        figures['heatmap'] = fig4
        if save_html:
            output_path = RESULTS_DIR / "13_bertopic_heatmap.html"
            fig4.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create heatmap visualization: {e}")

    # 5. Document visualization (if documents are available)
    try:
        fig5 = model.visualize_documents(docs, reduced_embeddings=model.umap_model.transform(model.embedding_model.embed(docs)))
        figures['documents'] = fig5
        if save_html:
            output_path = RESULTS_DIR / "14_bertopic_documents.html"
            fig5.write_html(str(output_path))
            print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"⚠ Could not create documents visualization: {e}")

    return figures


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_visualization_pipeline():
    """
    Run complete visualization pipeline for all 3 primary objectives.

    Generates:
    - BoW vs TF-IDF comparison visualizations
    - LDA predominant topics analysis (ALL topics)
    - BERTopic predominant topics analysis (ALL topics)

    All outputs saved to RESULTS_DIR.
    """
    print("=" * 80)
    print("TOPIC MODELING VISUALIZATION PIPELINE")
    print("=" * 80)

    # Objective 1: BoW vs TF-IDF Comparison
    print("\n[1/3] Comparing BoW vs TF-IDF Vectorization...")
    print("-" * 80)
    create_vectorization_comparison_summary()
    compare_vectorization_coherence()
    compare_k_selection()
    compare_topic_balance()

    # Objective 2: LDA Predominant Topics
    print("\n[2/3] Extracting ALL LDA Predominant Topics...")
    print("-" * 80)
    extract_lda_all_topics(model_type='bow')
    extract_lda_all_topics(model_type='tfidf')
    visualize_lda_all_topics_predominance(model_type='bow')
    visualize_lda_all_topics_predominance(model_type='tfidf')

    # Objective 3: BERTopic Predominant Topics
    print("\n[3/3] Extracting ALL BERTopic Predominant Topics...")
    print("-" * 80)
    extract_bertopic_all_topics()
    visualize_bertopic_all_topics_predominance()

    # Additional: Interactive Visualizations
    print("\n[BONUS] Generating Interactive Visualizations...")
    print("-" * 80)
    print("pyLDAvis (BoW)...")
    visualize_lda_pyldavis(model_type='bow')
    print("pyLDAvis (TF-IDF)...")
    visualize_lda_pyldavis(model_type='tfidf')
    print("BERTopic interactive plots...")
    visualize_bertopic_interactive_all()

    print("\n" + "=" * 80)
    print(f"✓ COMPLETE! All visualizations saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_visualization_pipeline()