"""
llm_topic_labeling.py
---------------------

Generates LLM-based topic labels for trained LDA and BERTopic models.

This script loads pure topic models (without LLM labeling) and generates
human-readable topic names using OpenAI's GPT models. It supports:
- German topic labels (original language)
- English topic labels (translation)
- Keyword translation for English visualizations (cosmetic)

The script is separate from topic_modeling.py to allow:
- Regenerating labels without retraining models
- Experimenting with different prompts/LLMs
- Independent topic modeling from labeling

Architecture
------------
1. topic_modeling.py → Generates pure topic models (c-TF-IDF keywords)
2. llm_topic_labeling.py → Adds LLM-generated labels (this script)
   - Uses TOPIC_LABELING_PROMPT_DE from topic_labeling.py (single source of truth)
   - LDA: Calls label_lda_topics_with_llm() which uses the prompt internally
   - BERTopic: Directly uses TOPIC_LABELING_PROMPT_DE in BERTopicOpenAI representation model
3. visualization.py → Creates visualizations from labeled artifacts

Inputs
------
- Trained LDA models from topic_models/lda/
- Trained BERTopic model from topic_models/bertopic/
- Original complaint texts for context

Outputs
-------
- German labeled artifacts → topic_models/*/german_labeled/
- English labeled artifacts → topic_models/*/english_labeled/

How to Run
----------
Default behavior: Generates BOTH German and English labels for both LDA and BERTopic.
Use --german-only flag to generate only German labels.

1) German + English labels (both models) - DEFAULT:

   python -m src.llm_topic_labeling

2) German labels only (both models):

   python -m src.llm_topic_labeling --german-only

3) Label only LDA models (German + English):

   python -m src.llm_topic_labeling --lda

4) Label only LDA models (German only):

   python -m src.llm_topic_labeling --lda --german-only

5) Label only BERTopic (German + English):

   python -m src.llm_topic_labeling --bertopic

6) Label only BERTopic (German only):

   python -m src.llm_topic_labeling --bertopic --german-only

Prerequisites
-------------
- Run topic_modeling.py first to generate base models
- Set OPENAI_API_KEY environment variable (in .env file)
- Install python-dotenv and openai packages

Cost Estimate
-------------
- LDA (5-7 topics): ~$0.05-0.10 per language
- BERTopic (6-8 topics): ~$0.05-0.10 per language
- Total (both models, bilingual): ~$0.20-0.40

Notes
-----
- Labels are generated based on top keywords + representative documents
- English keywords are translated for cosmetic visualization purposes
- The underlying topic models remain unchanged
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from gensim.models import LdaModel
from gensim import matutils
from bertopic import BERTopic
from bertopic.representation import OpenAI as BERTopicOpenAI
import openai

from src.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_NUM_REPRESENTATIVE_DOCS,
    LDA_TOPIC_DIR,
    LDA_TOPIC_DIR_DE,
    LDA_TOPIC_DIR_EN,
    BERTOPIC_TOPIC_DIR,
    BERTOPIC_TOPIC_DIR_DE,
    BERTOPIC_TOPIC_DIR_EN,
)
from src.data_loader import (
    load_lda_bow_model,
    load_lda_tfidf_model,
    load_lda_vectorized_artifacts,
    load_bertopic_model,
    load_cleaned_complaints,
    load_cleaned_for_bertopic,
)
from src.topic_labeling import (
    label_lda_topics_with_llm,
    translate_text_with_llm,
    get_high_probability_docs_for_lda,
    TOPIC_LABELING_PROMPT_DE,
)


# =========================================================================
# Helper Functions for Saving Labeled Artifacts
# =========================================================================

def _save_lda_labeled_artifacts(
    lda_model: LdaModel,
    topic_labels: dict,
    out_prefix: str,
    output_dir: Path,
    language: str = 'de',
    translate_keywords: bool = False,
    corpus=None,
    original_texts: List[str] = None
) -> None:
    """
    Save LDA artifacts with LLM-generated topic labels.

    Parameters
    ----------
    lda_model : LdaModel
        Trained Gensim LDA model
    topic_labels : dict
        Mapping of topic_id -> topic_name (from LLM)
    out_prefix : str
        Prefix for output files (e.g., "lda_bow" or "lda_tfidf")
    output_dir : Path
        Directory to save labeled artifacts
    language : str, default='de'
        Language code ('de' or 'en')
    translate_keywords : bool, default=False
        If True and language='en', translate keywords to English
    corpus : iterable, optional
        Gensim corpus (BoW format). Required if translate_keywords=True for context-aware translation.
    original_texts : List[str], optional
        Original complaint texts. Required if translate_keywords=True for context-aware translation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '' if language == 'de' else '_en'

    # Save topic-word table with labels
    topics = lda_model.show_topics(
        num_topics=lda_model.num_topics,
        num_words=15,
        formatted=False
    )
    rows = []
    for topic_id, word_probs in topics:
        topic_name = topic_labels.get(topic_id, f"Topic {topic_id}")
        for word, prob in word_probs:
            rows.append({
                "topic_id": topic_id,
                "topic_name": topic_name,
                "word": word,
                "weight": float(prob)
            })
    topics_df = pd.DataFrame(rows)

    # Translate keywords for English version (cosmetic only)
    if translate_keywords and language == 'en':
        print(f"  Translating keywords for {out_prefix}...")
        word_translations = {}

        # Topic-aware translation: use representative docs for context
        if corpus is not None and original_texts is not None:
            print(f"    Using topic-aware translation with representative documents...")
            corpus_list = list(corpus)
            unique_topic_ids = topics_df['topic_id'].unique()

            for topic_id in unique_topic_ids:
                # Get keywords for this topic
                topic_keywords = topics_df[topics_df['topic_id'] == topic_id]['word'].tolist()

                # Get representative documents for context
                try:
                    representative_docs = get_high_probability_docs_for_lda(
                        lda_model=lda_model,
                        corpus=corpus_list,
                        original_texts=original_texts,
                        topic_id=topic_id,
                        n_docs=3,  # Get 3 representative docs for context
                    )
                except Exception as e:
                    print(f"    Warning: Could not get representative docs for topic {topic_id}: {e}")
                    representative_docs = []

                # Translate keywords for this topic with context
                german_keywords = ', '.join(topic_keywords)
                try:
                    english_keywords = translate_text_with_llm(
                        german_keywords,
                        context_docs=representative_docs
                    )
                    english_words = [w.strip() for w in english_keywords.split(',')]

                    # Validate count
                    if len(english_words) != len(topic_keywords):
                        print(f"    Warning: Topic {topic_id} translation count mismatch "
                              f"(got {len(english_words)}, expected {len(topic_keywords)})")
                        # Fallback: keep German
                        for ger in topic_keywords:
                            word_translations[ger] = ger
                    else:
                        # Map translations
                        for ger, eng in zip(topic_keywords, english_words):
                            word_translations[ger] = eng
                        print(f"    Topic {topic_id}: Translated {len(topic_keywords)} keywords")

                except Exception as e:
                    print(f"    Warning: Translation failed for topic {topic_id}: {e}")
                    for ger in topic_keywords:
                        word_translations[ger] = ger

        else:
            # Fallback: batch translation without context (old method)
            print(f"    Warning: corpus/texts not provided, using translation without context")
            unique_words = topics_df['word'].unique()
            batch_size = 20

            for i in range(0, len(unique_words), batch_size):
                batch = unique_words[i:i+batch_size]
                german_batch = ', '.join(batch)
                try:
                    english_batch = translate_text_with_llm(german_batch)
                    english_words = [w.strip() for w in english_batch.split(',')]

                    if len(english_words) != len(batch):
                        print(f"    Warning: Translation count mismatch "
                              f"(got {len(english_words)}, expected {len(batch)})")
                        for ger in batch:
                            word_translations[ger] = ger
                    else:
                        for ger, eng in zip(batch, english_words):
                            word_translations[ger] = eng
                except Exception as e:
                    print(f"    Warning: Translation failed for batch: {e}")
                    for ger in batch:
                        word_translations[ger] = ger

        topics_df['word'] = topics_df['word'].map(
            lambda w: word_translations.get(w, w)
        )

    topics_df.to_csv(
        output_dir / f"{out_prefix}_topics_labeled{suffix}.csv",
        index=False
    )

    # Load and add labels to doc-topic distributions
    doc_topics_path = LDA_TOPIC_DIR / f"{out_prefix}_doc_topic_distributions.csv"
    if doc_topics_path.exists():
        doc_topics_df = pd.read_csv(doc_topics_path)
        doc_topics_df["dominant_topic_name"] = doc_topics_df["dominant_topic"].map(
            lambda tid: topic_labels.get(tid, f"Topic {tid}")
        )
        doc_topics_df.to_csv(
            output_dir / f"{out_prefix}_doc_topics_labeled{suffix}.csv",
            index=False
        )


def _save_bertopic_labeled_artifacts(
    topic_model: BERTopic,
    output_dir: Path,
    language: str = 'de',
    translate_keywords: bool = False
) -> None:
    """
    Save BERTopic artifacts with LLM-generated topic labels.

    Parameters
    ----------
    topic_model : BERTopic
        BERTopic model with LLM-generated topic names
    output_dir : Path
        Directory to save labeled artifacts
    language : str, default='de'
        Language code ('de' or 'en')
    translate_keywords : bool, default=False
        If True and language='en', translate keywords in Representation field
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '' if language == 'de' else '_en'

    # Save model (without representation_model to avoid pickle errors)
    temp_repr_model = topic_model.representation_model
    topic_model.representation_model = None
    topic_model.save(output_dir / "bertopic_model")
    topic_model.representation_model = temp_repr_model

    # Save topic info with LLM labels
    info_df = topic_model.get_topic_info()

    # Translate keywords in Representation field (cosmetic only)
    if translate_keywords and language == 'en':
        print("  Translating BERTopic keywords...")
        translated_representations = []

        for _, row in info_df.iterrows():
            try:
                keywords = eval(row['Representation']) \
                    if isinstance(row['Representation'], str) \
                    else row['Representation']

                if isinstance(keywords, list) and len(keywords) > 0:
                    german_keywords = ', '.join(keywords[:10])
                    english_keywords = translate_text_with_llm(german_keywords)
                    translated_keywords = [
                        w.strip() for w in english_keywords.split(',')
                    ]
                    translated_representations.append(translated_keywords)
                else:
                    translated_representations.append(keywords)
            except Exception as e:
                print(f"    Warning: Translation failed for topic {row['Topic']}: {e}")
                translated_representations.append(row['Representation'])

        info_df['Representation'] = translated_representations

    info_df.to_csv(
        output_dir / f"bertopic_topic_info_labeled{suffix}.csv",
        index=False
    )

    # Save doc-topics with topic names
    doc_topics = pd.read_csv(BERTOPIC_TOPIC_DIR / "bertopic_doc_topics.csv")
    topic_names = {row["Topic"]: row["Name"] for _, row in info_df.iterrows()}
    doc_topics["topic_name"] = doc_topics["topic_id"].map(topic_names)
    doc_topics.to_csv(
        output_dir / f"bertopic_doc_topics_labeled{suffix}.csv",
        index=False
    )


def _save_bertopic_labeled_artifacts_with_keywords(
    topic_model: BERTopic,
    output_dir: Path,
    original_keywords: dict,
    language: str = 'de',
    translate_keywords: bool = False,
    translated_labels: dict = None,
    original_keywords_with_scores: dict = None
) -> None:
    """
    Save BERTopic artifacts with LLM-generated topic labels AND original c-TF-IDF keywords.

    Parameters
    ----------
    topic_model : BERTopic
        BERTopic model with LLM-generated topic names
    output_dir : Path
        Directory to save labeled artifacts
    original_keywords : dict
        Mapping of topic_id -> list of c-TF-IDF keywords (preserved from original model)
    language : str, default='de'
        Language code ('de' or 'en')
    translate_keywords : bool, default=False
        If True and language='en', translate keywords in Representation field
    translated_labels : dict, optional
        Mapping of topic_id -> translated_label (English). If provided and language='en',
        replaces the Name column with translated topic names.
    original_keywords_with_scores : dict, optional
        Mapping of topic_id -> list of (keyword, score) tuples for restoring topic_representations_
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '' if language == 'de' else '_en'

    # Update model's internal topic labels with custom labels
    # This ensures interactive visualizations (visualize_topics, visualize_hierarchy, etc.)
    # display the LLM-generated labels when custom_labels=True is passed
    # IMPORTANT: Use set_topic_labels() to properly convert dict to list format
    if translated_labels:
        topic_model.set_topic_labels(translated_labels)

    # Save topic info with LLM labels AND original c-TF-IDF keywords
    info_df = topic_model.get_topic_info()

    # Load corrected counts from the CSV saved by topic_modeling.py
    # The model object has old counts (before outlier reduction), but the CSV has correct counts
    corrected_topic_info_path = BERTOPIC_TOPIC_DIR / "bertopic_topic_info.csv"
    if corrected_topic_info_path.exists():
        corrected_counts = pd.read_csv(corrected_topic_info_path)[['Topic', 'Count']]
        # Update Count column with corrected values
        info_df = info_df.drop(columns=['Count']).merge(corrected_counts, on='Topic', how='left')

    # Replace Representation field with original c-TF-IDF keywords
    representations = []
    translated_keywords_dict = {}
    for _, row in info_df.iterrows():
        topic_id = row['Topic']
        if topic_id in original_keywords:
            keywords = original_keywords[topic_id]
        else:
            # Fallback to current representation
            keywords = eval(row['Representation']) if isinstance(row['Representation'], str) else row['Representation']

        # Translate keywords for English version
        if translate_keywords and language == 'en' and isinstance(keywords, list):
            try:
                german_keywords = ', '.join(keywords[:10])
                english_keywords = translate_text_with_llm(german_keywords)
                translated_keywords = [w.strip() for w in english_keywords.split(',')]
                representations.append(translated_keywords)
                if topic_id != -1:
                    translated_keywords_dict[topic_id] = translated_keywords
            except Exception as e:
                print(f"    Warning: Translation failed for topic {topic_id}: {e}")
                representations.append(keywords)
        else:
            representations.append(keywords)

    info_df['Representation'] = representations

    # Update Name column with translated labels if provided
    if translated_labels and language == 'en':
        info_df['Name'] = info_df['Topic'].map(translated_labels).fillna(info_df['Name'])

    # Restore c-TF-IDF keywords to topic_representations_ for barchart visualizations
    if original_keywords_with_scores:
        new_topic_representations = {}
        for topic_id, keywords_with_scores in original_keywords_with_scores.items():
            if not keywords_with_scores or topic_id == -1:
                continue

            if translate_keywords and language == 'en' and topic_id in translated_keywords_dict:
                english_words = translated_keywords_dict[topic_id]
                expected_len = len(keywords_with_scores)
                actual_len = len(english_words)

                if actual_len == expected_len:
                    # Perfect match - use all translated keywords
                    new_topic_representations[topic_id] = [
                        (eng_word, score) for eng_word, (_, score) in zip(english_words, keywords_with_scores)
                    ]
                elif actual_len > expected_len:
                    # More translated keywords than expected - truncate to match
                    print(f"    ⚠ Topic {topic_id}: LLM returned {actual_len} keywords, expected {expected_len}. Truncating.")
                    new_topic_representations[topic_id] = [
                        (english_words[i], score) for i, (_, score) in enumerate(keywords_with_scores)
                    ]
                else:
                    # Fewer translated keywords than expected - pad with German keywords
                    print(f"    ⚠ Topic {topic_id}: LLM returned {actual_len} keywords, expected {expected_len}. Padding with German.")
                    mixed_keywords = []
                    for i, (german_word, score) in enumerate(keywords_with_scores):
                        if i < actual_len:
                            mixed_keywords.append((english_words[i], score))
                        else:
                            mixed_keywords.append((german_word, score))
                    new_topic_representations[topic_id] = mixed_keywords
            else:
                new_topic_representations[topic_id] = keywords_with_scores

        topic_model.topic_representations_ = new_topic_representations

    # Save model (without representation_model to avoid pickle errors)
    temp_repr_model = topic_model.representation_model
    topic_model.representation_model = None
    topic_model.save(output_dir / "bertopic_model")
    topic_model.representation_model = temp_repr_model

    info_df.to_csv(
        output_dir / f"bertopic_topic_info_labeled{suffix}.csv",
        index=False
    )

    # Save doc-topics with topic names
    doc_topics = pd.read_csv(BERTOPIC_TOPIC_DIR / "bertopic_doc_topics.csv")
    topic_names = {row["Topic"]: row["Name"] for _, row in info_df.iterrows()}
    doc_topics["topic_name"] = doc_topics["topic_id"].map(topic_names)
    doc_topics.to_csv(
        output_dir / f"bertopic_doc_topics_labeled{suffix}.csv",
        index=False
    )


# =========================================================================
# Main Labeling Pipeline
# =========================================================================

def label_lda_models(generate_english: bool = False) -> None:
    """
    Generate LLM-based labels for LDA models (BoW and TF-IDF).

    Parameters
    ----------
    generate_english : bool, default=False
        If True, also generate English translations of labels and keywords
    """
    print("\n" + "="*70)
    print("GENERATING LLM LABELS FOR LDA MODELS")
    print("="*70)

    # Load artifacts
    artifacts = load_lda_vectorized_artifacts()
    X_bow = artifacts["X_bow"]
    X_tfidf = artifacts["X_tfidf"]
    bow_vocab = artifacts["bow_vocab"]
    tfidf_vocab = artifacts["tfidf_vocab"]

    # Load original texts for context
    df_cleaned = load_cleaned_complaints()
    original_texts = df_cleaned["description"].fillna("").astype(str).tolist()

    # Load trained models
    lda_bow = load_lda_bow_model()
    lda_tfidf = load_lda_tfidf_model()

    # Convert sparse matrices to corpus
    corpus_bow = matutils.Sparse2Corpus(X_bow, documents_columns=False)
    corpus_tfidf = matutils.Sparse2Corpus(X_tfidf, documents_columns=False)

    # ========== GERMAN LABELS ==========
    print("\n[1/2] Generating German labels...")
    print("-" * 70)

    # BoW German
    print("\n→ LDA BoW (German)")
    labels_bow_de = label_lda_topics_with_llm(
        lda_model=lda_bow,
        corpus=corpus_bow,
        original_texts=original_texts,
        vocab=bow_vocab,
    )

    # TF-IDF German
    print("\n→ LDA TF-IDF (German)")
    labels_tfidf_de = label_lda_topics_with_llm(
        lda_model=lda_tfidf,
        corpus=corpus_tfidf,
        original_texts=original_texts,
        vocab=tfidf_vocab,
    )

    # Save German artifacts
    _save_lda_labeled_artifacts(
        lda_bow, labels_bow_de, "lda_bow",
        LDA_TOPIC_DIR_DE, language='de'
    )
    _save_lda_labeled_artifacts(
        lda_tfidf, labels_tfidf_de, "lda_tfidf",
        LDA_TOPIC_DIR_DE, language='de'
    )

    print(f"\n✅ Saved German labeled artifacts to: {LDA_TOPIC_DIR_DE}")

    # ========== ENGLISH LABELS ==========
    if generate_english:
        print("\n[2/2] Translating labels to English...")
        print("-" * 70)

        # Translate BoW German labels to English
        print("\n→ LDA BoW (Translating German labels)")
        labels_bow_en = {}
        for topic_id, german_label in labels_bow_de.items():
            english_label = translate_text_with_llm(german_label)
            labels_bow_en[topic_id] = english_label
            print(f"   Topic {topic_id}: {german_label} → {english_label}")

        # Translate TF-IDF German labels to English
        print("\n→ LDA TF-IDF (Translating German labels)")
        labels_tfidf_en = {}
        for topic_id, german_label in labels_tfidf_de.items():
            english_label = translate_text_with_llm(german_label)
            labels_tfidf_en[topic_id] = english_label
            print(f"   Topic {topic_id}: {german_label} → {english_label}")

        # Save English artifacts (with keyword translation)
        _save_lda_labeled_artifacts(
            lda_bow, labels_bow_en, "lda_bow",
            LDA_TOPIC_DIR_EN, language='en', translate_keywords=True,
            corpus=corpus_bow, original_texts=original_texts
        )
        _save_lda_labeled_artifacts(
            lda_tfidf, labels_tfidf_en, "lda_tfidf",
            LDA_TOPIC_DIR_EN, language='en', translate_keywords=True,
            corpus=corpus_tfidf, original_texts=original_texts
        )

        print(f"\n✅ Saved English labeled artifacts to: {LDA_TOPIC_DIR_EN}")


def label_bertopic_model(generate_english: bool = False) -> None:
    """
    Generate LLM-based labels for BERTopic model.

    Parameters
    ----------
    generate_english : bool, default=False
        If True, also generate English translations of labels and keywords
    """
    print("\n" + "="*70)
    print("GENERATING LLM LABELS FOR BERTOPIC MODEL")
    print("="*70)

    # Load pure model and texts
    topic_model = load_bertopic_model()
    df_bertopic = load_cleaned_for_bertopic()
    texts = df_bertopic["bertopic_description"].fillna("").astype(str).tolist()

    # Preserve original c-TF-IDF keywords AND scores before calling update_topics()
    # update_topics() replaces them with LLM labels, but we want to keep the original keywords
    original_topic_info = topic_model.get_topic_info()
    original_keywords = {}
    original_keywords_with_scores = {}  # Preserve scores for topic_representations_
    for _, row in original_topic_info.iterrows():
        topic_id = row['Topic']
        # Get keywords from the topic model directly
        if topic_id != -1:
            topic_keywords = topic_model.get_topic(topic_id)
            if topic_keywords:
                # Store keywords only (for CSV)
                original_keywords[topic_id] = [word for word, _ in topic_keywords[:10]]
                # Store keywords with scores (for topic_representations_)
                original_keywords_with_scores[topic_id] = topic_keywords[:10]
        else:
            # For outlier topic (-1), use Representation field
            original_keywords[topic_id] = eval(row['Representation']) if isinstance(row['Representation'], str) else row['Representation']
            original_keywords_with_scores[topic_id] = []

    # ========== GERMAN LABELS ==========
    print("\n[1/2] Generating German labels...")
    print("-" * 70)

    # Create German representation model
    # Use the canonical prompt template from topic_labeling.py (single source of truth)
    representation_model_de = BERTopicOpenAI(
        client=openai.OpenAI(api_key=OPENAI_API_KEY),
        model=LLM_MODEL,
        delay_in_seconds=0.5,
        chat=True,
        prompt=TOPIC_LABELING_PROMPT_DE,
        nr_docs=LLM_NUM_REPRESENTATIVE_DOCS,  # Send 6 representative docs to LLM (increased from default 4)
        doc_length=400,  # Limit each doc to 400 chars to reduce noise and focus on core complaint
        tokenizer="char",  # Truncate by character count (required when doc_length is specified)
    )

    # Update topic representations with German labels
    print("\n→ Updating BERTopic with German labels...")
    topic_model.update_topics(texts, representation_model=representation_model_de)

    # Get German topic names from the updated model
    german_topic_info = topic_model.get_topic_info()
    german_labels = {row['Topic']: row['Name'] for _, row in german_topic_info.iterrows()}

    print(f"\n   Generated German labels:")
    for topic_id, label in german_labels.items():
        if topic_id != -1:
            print(f"   Topic {topic_id}: {label}")

    # Save German artifacts with ORIGINAL keywords restored and German labels
    _save_bertopic_labeled_artifacts_with_keywords(
        topic_model, BERTOPIC_TOPIC_DIR_DE, original_keywords,
        language='de', translated_labels=german_labels,
        original_keywords_with_scores=original_keywords_with_scores
    )

    print(f"\n✅ Saved German labeled artifacts to: {BERTOPIC_TOPIC_DIR_DE}")

    # ========== ENGLISH LABELS ==========
    if generate_english:
        print("\n[2/2] Translating labels to English...")
        print("-" * 70)

        # Translate German labels to English (don't regenerate with LLM)
        print("\n→ Translating German labels...")
        english_labels = {}
        for topic_id, german_label in german_labels.items():
            # BERTopic adds "{topic_id}_" prefix to names - strip it before translating
            if '_' in german_label and german_label.split('_')[0].lstrip('-').isdigit():
                # Extract prefix and actual label
                parts = german_label.split('_', 1)
                prefix = parts[0]
                label_without_prefix = parts[1] if len(parts) > 1 else german_label
                # Translate just the label part
                translated_label = translate_text_with_llm(label_without_prefix)
                # Add prefix back
                english_label = f"{prefix}_{translated_label}"
            else:
                # No prefix, translate as-is
                english_label = translate_text_with_llm(german_label)

            english_labels[topic_id] = english_label
            if topic_id != -1:
                print(f"   Topic {topic_id}: {german_label} → {english_label}")

        # Manually update the topic names with English labels
        # We don't call update_topics() again to avoid regenerating labels
        for topic_id, english_label in english_labels.items():
            topic_model.topic_labels_[topic_id] = english_label

        # Update custom_labels_ if it exists
        if hasattr(topic_model, 'custom_labels_') and topic_model.custom_labels_:
            topic_model.custom_labels_ = english_labels

        # Save English artifacts with ORIGINAL keywords restored and translated
        _save_bertopic_labeled_artifacts_with_keywords(
            topic_model, BERTOPIC_TOPIC_DIR_EN, original_keywords,
            language='en', translate_keywords=True, translated_labels=english_labels,
            original_keywords_with_scores=original_keywords_with_scores
        )

        print(f"\n✅ Saved English labeled artifacts to: {BERTOPIC_TOPIC_DIR_EN}")


def run_labeling_pipeline(
    label_lda: bool = True,
    label_bertopic: bool = True,
    generate_english: bool = True
) -> None:
    """
    Run complete LLM topic labeling pipeline.

    Parameters
    ----------
    label_lda : bool, default=True
        If True, generate labels for LDA models
    label_bertopic : bool, default=True
        If True, generate labels for BERTopic model
    generate_english : bool, default=True
        If True, generate English translations in addition to German labels.
        Default is True (generates both languages). Use --german-only flag
        from CLI to generate only German labels.
    """
    # Check API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "":
        raise ValueError(
            "OPENAI_API_KEY not set. Please set it in your .env file:\n"
            "OPENAI_API_KEY=sk-your-api-key-here"
        )

    print("\n" + "="*70)
    print("LLM TOPIC LABELING PIPELINE")
    print("="*70)
    print(f"\nLLM Model: {LLM_MODEL}")
    print(f"Languages: German" + (" + English" if generate_english else ""))
    print(f"Models: " + (
        "LDA + BERTopic" if (label_lda and label_bertopic)
        else "LDA only" if label_lda
        else "BERTopic only"
    ))

    # Run labeling
    if label_lda:
        label_lda_models(generate_english=generate_english)

    if label_bertopic:
        label_bertopic_model(generate_english=generate_english)

    # Summary
    print("\n" + "="*70)
    print("✅ LABELING COMPLETE!")
    print("="*70)
    print("\nGenerated artifacts:")
    if label_lda:
        print(f"  - LDA German: {LDA_TOPIC_DIR_DE}")
        if generate_english:
            print(f"  - LDA English: {LDA_TOPIC_DIR_EN}")
    if label_bertopic:
        print(f"  - BERTopic German: {BERTOPIC_TOPIC_DIR_DE}")
        if generate_english:
            print(f"  - BERTopic English: {BERTOPIC_TOPIC_DIR_EN}")

    print("\nNext step:")
    print("  Run visualizations:")
    if generate_english:
        print("    python -m src.visualization --german --english")
    else:
        print("    python -m src.visualization --german")
    print("="*70 + "\n")


# =========================================================================
# Command-Line Interface
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LLM-based topic labels for trained topic models"
    )

    parser.add_argument(
        "--lda",
        action="store_true",
        help="Generate labels for LDA models (BoW and TF-IDF)"
    )
    parser.add_argument(
        "--bertopic",
        action="store_true",
        help="Generate labels for BERTopic model"
    )
    parser.add_argument(
        "--german-only",
        action="store_true",
        help="Generate German labels only (default: generates both German and English)"
    )

    args = parser.parse_args()

    # Default: label both LDA and BERTopic if no flags specified
    label_lda = args.lda or (not args.lda and not args.bertopic)
    label_bertopic = args.bertopic or (not args.lda and not args.bertopic)

    run_labeling_pipeline(
        label_lda=label_lda,
        label_bertopic=label_bertopic,
        generate_english=not args.german_only
    )
