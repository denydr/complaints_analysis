"""
topic_labeling.py
-----------------

LLM-based topic labeling helpers for LDA and BERTopic models.

This module provides functionality to generate human-readable German topic names
using Large Language Models (LLMs) based on:
- Representative keywords from topic models
- Sample documents with high topic relevance

Architecture
------------
- This module generates GERMAN topic labels only
- English labels are created by translating German labels (see translate_text_with_llm)
- This ensures semantic consistency between German and English versions
- TOPIC_LABELING_PROMPT_DE constant defines the canonical prompt (single source of truth)
  used by both LDA (via call_openai_for_label) and BERTopic (via llm_topic_labeling.py)

Constants
---------
- TOPIC_LABELING_PROMPT_DE: Canonical German topic labeling prompt template

Functions
---------
- label_lda_topics_with_llm(): Generate German topic labels for LDA topics
- get_high_probability_docs_for_lda(): Extract representative documents for LDA topics
- call_openai_for_label(): Core LLM API call for German label generation
- translate_text_with_llm(): Translate German text to English using LLM

Usage
-----
For LDA (German labels):
    from src.topic_labeling import label_lda_topics_with_llm
    from gensim.models import LdaModel

    lda = LdaModel.load("path/to/model.gensim")
    labels_de = label_lda_topics_with_llm(
        lda_model=lda,
        corpus=corpus,
        original_texts=texts,
        vocab=vocab,
    )

For translation to English:
    from src.topic_labeling import translate_text_with_llm

    labels_en = {
        topic_id: translate_text_with_llm(german_label)
        for topic_id, german_label in labels_de.items()
    }

For BERTopic:
    Import and use TOPIC_LABELING_PROMPT_DE in llm_topic_labeling.py
    with BERTopic's OpenAI representation model
"""

import numpy as np
import openai
from typing import List, Dict, Tuple
from gensim.models import LdaModel

from src.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_NUM_KEYWORDS,
    LLM_NUM_REPRESENTATIVE_DOCS,
)


# =======================
# Prompt Templates
# =======================

# German topic labeling prompt (canonical version)
# Uses BERTopic placeholder format: [KEYWORDS] and [DOCUMENTS]
# For LDA usage, replace placeholders with actual formatted values
TOPIC_LABELING_PROMPT_DE = """You are analyzing topics from Munich city complaint data (in German).

Keywords from topic model: [KEYWORDS]

Representative complaint examples (German text):
[DOCUMENTS]

Task: Generate a concise German topic label (3-5 words) that accurately describes this topic cluster.

Quality criteria:
- DESCRIPTIVE: Capture the main theme that distinguishes this topic from others
- CONCRETE: Use specific terms rather than vague generalizations
- GROUNDED: Prefer terminology that actually appears in the complaint texts
- STANDALONE: Label should be understandable without additional context

STRICT EVIDENCE RULES:
- Your label must ONLY use information explicitly stated in the keywords and examples
- When possible, reuse exact wording from the keywords or examples rather than paraphrasing
- Do NOT infer causes, intent, severity, or legality not present in the text
- Do NOT interpret one word as another (e.g., "kaputt" ≠ "gefährlich", "alt" ≠ "unsicher")
- If the evidence is ambiguous, describe what is explicitly mentioned, not what it might mean

Avoid:
- Overly generic terms that could apply to many topics
- Meta-references (e.g., "Thema über...", "Beschwerden bezüglich...")
- Normative judgments not explicitly stated in the data

Output only the German label:"""


def call_openai_for_label(
    keywords: List[str],
    example_docs: List[str],
    task: str = 'label'
) -> str:
    """
    Call OpenAI API to generate a German topic label or translation.

    Parameters
    ----------
    keywords : List[str]
        Top words representing the topic (e.g., ["mast", "lampe", "leuchtet"])
    example_docs : List[str]
        Sample documents that exemplify the topic (1-3 sentences each)
    task : str, default='label'
        Task type: 'label' for German topic labeling, 'translate' for German→English translation

    Returns
    -------
    str
        Generated topic label (e.g., "Defekte Straßenbeleuchtung") or English translation

    Raises
    ------
    ValueError
        If OPENAI_API_KEY is not set
    openai.OpenAIError
        If API call fails
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not set. Please set the environment variable:\n"
            "export OPENAI_API_KEY='sk-...'"
        )

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Construct prompt based on task
    if task == 'label':
        # Use the canonical prompt template, replacing BERTopic placeholders with actual values
        keywords_str = ', '.join(keywords)
        documents_str = chr(10).join(f"{i+1}. {doc[:200]}" for i, doc in enumerate(example_docs))

        prompt = TOPIC_LABELING_PROMPT_DE.replace('[KEYWORDS]', keywords_str).replace('[DOCUMENTS]', documents_str)

    elif task == "translate":
        text_to_translate = ", ".join(keywords)

        # Build prompt with optional representative documents for context
        if example_docs and len(example_docs) > 0:
            docs_context = "\n".join(f"{i+1}. {doc[:150]}" for i, doc in enumerate(example_docs[:3]))
            prompt = f"""Translate the following German keywords into English.

Context: Munich city municipal infrastructure complaints.

Keywords to translate: {text_to_translate}

Representative complaint examples (showing how these keywords are used):
{docs_context}

Rules:
- Output ONLY the translated keywords as a comma-separated list, nothing else.
- Use the examples above to understand the correct meaning of ambiguous words.
- Preserve the exact same number of keywords (if input has 5 words, output must have 5 words).
- Keep the same order as the input.

English keywords:"""
        else:
            prompt = f"""Translate the following German text into English.

Context: Munich city municipal infrastructure complaints.

Rules:
- Output ONLY the translation, nothing else (no explanations, no quotes, no prefixes).
- Preserve the original format exactly (keep commas if present, same number of comma-separated items).
- Do not paraphrase or expand; translate as a short phrase suitable as a topic label.

German:
{text_to_translate}

English:"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # More deterministic for consistency
            max_tokens=50,     # Topic labels should be short
        )

        label = response.choices[0].message.content.strip()

        # Clean up common artifacts
        label = label.strip('"').strip("'").strip()

        return label

    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        # Fallback: return first few keywords as label
        return " ".join(keywords[:3])


def get_high_probability_docs_for_lda(
    lda_model: LdaModel,
    corpus,
    original_texts: List[str],
    topic_id: int,
    n_docs: int = 2,
    min_probability: float = 0.5
) -> List[str]:
    """
    Extract documents with highest probability for a given LDA topic.

    LDA doesn't have "representative documents" like BERTopic, but we can
    find documents where this topic has high probability (> 0.5).

    Parameters
    ----------
    lda_model : gensim.models.LdaModel
        Trained LDA model
    corpus : iterable
        Gensim corpus (BoW format)
    original_texts : List[str]
        Original complaint texts aligned with corpus order
    topic_id : int
        Topic ID to find representative documents for
    n_docs : int, default=2
        Number of representative documents to return
    min_probability : float, default=0.5
        Minimum topic probability threshold for a document to be considered

    Returns
    -------
    List[str]
        List of up to n_docs representative documents
    """
    topic_probs = []

    # Get topic probability for all documents
    for doc_idx, bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        # Find probability for our topic
        prob = 0.0
        for tid, p in doc_topics:
            if tid == topic_id:
                prob = p
                break
        topic_probs.append((doc_idx, prob))

    # Sort by probability (descending) and filter by threshold
    topic_probs.sort(key=lambda x: x[1], reverse=True)
    high_prob_docs = [
        (idx, prob) for idx, prob in topic_probs
        if prob >= min_probability
    ]

    # If we don't have enough documents above threshold, take top n anyway
    if len(high_prob_docs) < n_docs:
        high_prob_docs = topic_probs[:n_docs]
    else:
        high_prob_docs = high_prob_docs[:n_docs]

    # Extract the actual text
    representative_docs = []
    for doc_idx, prob in high_prob_docs:
        if doc_idx < len(original_texts):
            representative_docs.append(original_texts[doc_idx])

    return representative_docs


def label_lda_topics_with_llm(
    lda_model: LdaModel,
    corpus,
    original_texts: List[str],
    vocab: List[str],
    num_keywords: int = None,
    num_docs: int = None,
) -> Dict[int, str]:
    """
    Generate German LLM-based topic labels for all topics in an LDA model.

    For each topic:
    1. Extract top N keywords (words with highest probability)
    2. Find N documents with highest topic probability
    3. Send to LLM to generate a concise German label

    Note: English labels are created by translating German labels separately.

    Parameters
    ----------
    lda_model : gensim.models.LdaModel
        Trained LDA model
    corpus : iterable
        Gensim corpus in BoW format (aligned with original_texts)
    original_texts : List[str]
        Original German complaint texts (not lemmatized, for LLM context)
    vocab : List[str]
        Vocabulary list from the vectorizer (aligned with LDA id2word)
    num_keywords : int, optional
        Number of top keywords to include. Defaults to LLM_NUM_KEYWORDS from config.
    num_docs : int, optional
        Number of representative documents to include. Defaults to LLM_NUM_REPRESENTATIVE_DOCS.

    Returns
    -------
    Dict[int, str]
        Mapping of topic_id -> German topic_label
        Example: {0: "Defekte Straßenbeleuchtung", 1: "Gefährliche Spielplätze", ...}

    Raises
    ------
    ValueError
        If OPENAI_API_KEY is not set or if inputs are misaligned
    """
    if num_keywords is None:
        num_keywords = LLM_NUM_KEYWORDS
    if num_docs is None:
        num_docs = LLM_NUM_REPRESENTATIVE_DOCS

    num_topics = lda_model.num_topics
    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_keywords, formatted=False)

    # Convert corpus to list for repeated iteration
    corpus_list = list(corpus)

    labels = {}

    print(f"\n{'='*60}")
    print(f"Generating German topic labels using {LLM_MODEL}...")
    print(f"{'='*60}\n")

    for topic_id, word_probs in topics:
        # Extract top keywords
        keywords = [word for word, prob in word_probs]

        # Get representative documents
        try:
            representative_docs = get_high_probability_docs_for_lda(
                lda_model=lda_model,
                corpus=corpus_list,
                original_texts=original_texts,
                topic_id=topic_id,
                n_docs=num_docs,
            )
        except Exception as e:
            print(f"Warning: Could not extract representative docs for topic {topic_id}: {e}")
            representative_docs = []

        # Call LLM to generate German label
        try:
            label = call_openai_for_label(
                keywords=keywords,
                example_docs=representative_docs if representative_docs else keywords[:3],
                task='label'
            )
            labels[topic_id] = label
            print(f"Topic {topic_id}: {label}")
            print(f"  Keywords: {', '.join(keywords[:5])}")
            if representative_docs:
                print(f"  Example: {representative_docs[0][:100]}...")
            print()

        except Exception as e:
            print(f"Error generating label for topic {topic_id}: {e}")
            # Fallback to keyword-based label
            labels[topic_id] = " ".join(keywords[:3])

    print(f"{'='*60}")
    print(f"Generated {len(labels)} topic labels")
    print(f"{'='*60}\n")

    return labels


def translate_text_with_llm(text: str, context_docs: List[str] = None) -> str:
    """
    Translate German text to English using LLM.

    Parameters
    ----------
    text : str
        German text to translate (can be a single word or comma-separated list)
    context_docs : List[str], optional
        Representative documents that provide semantic context for translation.
        Helps disambiguate words like "laufen" (walk vs. running) or "schloß" (castle vs. lock).

    Returns
    -------
    str
        English translation
    """
    if not text or text.strip() == "":
        return ""

    # For comma-separated keywords, split and translate
    keywords = [k.strip() for k in text.split(',')]

    try:
        result = call_openai_for_label(
            keywords=keywords,
            example_docs=context_docs if context_docs else [],
            task='translate'
        )

        # Basic cleanup: strip "English:" prefix if LLM adds it
        if ':' in result and result.split(':')[0].lower() in ['english', 'translation']:
            result = result.split(':', 1)[1].strip()

        # Capitalize first letter for topic names (multi-word phrases, not comma-separated keywords)
        if ',' not in result and result and not result[0].isupper():
            result = result[0].upper() + result[1:]

        return result
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original if translation fails
