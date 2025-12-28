# Munich Complaints Analysis

Topic modeling and analysis of Munich Open311 complaints using LDA and BERTopic with LLM-generated topic labels.

## Overview

This project implements a comprehensive topic modeling pipeline for analyzing German-language municipal complaints and 
identifying prevalent topics within Munich's Open311 system. It combines traditional probabilistic models (LDA) with modern
transformer-based approaches (BERTopic), enhanced by LLM-generated, human-readable topic labels.

> **Note on topic label variability:**  
> Although random seeds are fixed to ensure reproducibility of topic assignments,  
> LLM-generated topic labels may vary slightly between pipeline runs.  
> These variations are purely linguistic; the underlying topic structure and semantic meaning remain unchanged.

## Features

- **Dual Topic Modeling**: LDA (Bag-of-Words & TF-IDF) and BERTopic with transformer embeddings
- **Automated K Selection**: Coherence-based optimization for determining optimal number of topics for LDA
- **LLM-Generated Topic Labels**: Human-readable topic names using GPT-4o-mini via the OpenAI API
- **Bilingual Support**: Generate topic labels in both German (original) and English
- **Interactive Visualizations**: pyLDAvis for LDA, interactive HTML for BERTopic
- **Comprehensive Analysis**: Topic distributions and representative documents
- **Reproducible Pipeline**: Fixed random seeds and orchestrated workflow

## Setup

### 1. Clone the repository
```bash
git clone <https://github.com/denydr/complaints_analysis.git>
cd complaints_analysis
```
### 2. Install Python

Download version 3.12.0 and install it.

### 3. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```  

### 5. Configure API keys
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Get your API key from: https://platform.openai.com/api-keys
```

Your `.env` file should look like:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important**: Each user needs their own OpenAI API key. The `.env` file is gitignored to prevent accidentally committing secrets. Never commit your actual API key to version control.

### 6. Verify setup
```bash
# Check that your API key is loaded
python -c "from src.config import OPENAI_API_KEY; print('API key configured!' if OPENAI_API_KEY else 'API key missing')"
```

## Pipeline Architecture

The topic modeling pipeline consists of 5 sequential steps:

```
1. Data Cleaning          → Preprocesses raw German complaint texts
2. Vectorization (LDA)    → Creates BoW and TF-IDF matrices
3. Topic Modeling         → Trains LDA and/or BERTopic models
4. LLM Labeling (optional)→ Generates human-readable topic names
5. Visualization          → Creates interactive charts and analyses
```

Each step saves artifacts to disk, allowing incremental execution and experimentation.

## Usage

### Dataset Acquisition

**Selected Dataset:**  *Mach München Besser – Open311 GeoReport v2* dataset, published by the **City of Munich (Landeshauptstadt München)**. It contains anonymized **citizen complaints and issue reports** submitted via the city’s official Open311 API,  
including unstructured text fields (`description`) and structured metadata (`service_name`, `status`,`requested_datetime`, etc.).

**Source:** https://machmuenchenbesser.de/georeport/v2/requests.json

The dataset extraction is done via the jupyter notebook `notebooks/01_dataset_exploration.ipynb`.

> **Data reproducibility note:**  
> The API-based data acquisition step should not be re-executed, as repeated queries may return inconsistent sample counts over time.
> To ensure reproducibility, the dataset used in this project is included in the repository.  
> For details, see **Section (2) Dataset Sampling** in `notebooks/01_dataset_exploration.ipynb`.


### Running Full Pipeline (Recommended)

Run the complete pipeline using the `src.main` module, opting for one of the following pipeline variants:

```bash
# Full pipeline with bilingual labels
python -m src.main --clean --vectorize --train --label --visualize --german --english
```  

```bash
# Full pipeline, no LLM labels (faster, no API costs)
python -m src.main --clean --vectorize --train --visualize
```  

```bash
# LDA pipeline (no LLM labels)
python -m src.main --clean --vectorize --train --models lda --visualize 
```  

```bash
# LDA pipeline with bilingual labels
python -m src.main --clean --vectorize --train --models lda --label --visualize --german --english
```  

```bash
# BERTopic pipeline (no LLM labels)
python -m src.main --clean --vectorize --train --models bertopic --visualize 
```  

```bash  
# BERTopic pipeline with bilingual labels
python -m src.main --clean --train --models bertopic --label --visualize --german --english
```  

**Key Flags**:
- **Steps**: `--clean`, `--vectorize`, `--train`, `--label`, `--visualize` (specify which to run)
- **Models**: `--models {lda,bertopic,all}` (default: `all`)
- **Languages**: `--german`, `--english` (for LLM labels and visualizations)

*Note*: Steps must be explicitly specified and execute in order.

### Running Step-By-Step Pipeline

Run pipeline steps independently for more control:

```bash
# Step 1: Clean raw complaint texts
python -m src.cleaning
```  

```bash
# Step 2: Vectorize for LDA (BoW + TF-IDF)
python -m src.vectorization
```  

*Note:* As an optional step after cleaning and vectorization, sanity checkups can be executed within `notebooks/02_sanity_checkups.ipynb`. 

```bash
# Step 3: Train topic models
python -m src.topic_modeling              # Both LDA and BERTopic
python -m src.topic_modeling --lda        # LDA only
python -m src.topic_modeling --bertopic   # BERTopic only
```  

```bash
# Step 4: Generate LLM labels (optional)
python -m src.llm_topic_labeling                  # Both models, both languages
python -m src.llm_topic_labeling --german-only    # German only (~50% cost)
python -m src.llm_topic_labeling --lda            # LDA only
python -m src.llm_topic_labeling --bertopic       # BERTopic only
```  

```bash
# Step 5: Create visualizations
python -m src.visualization --german --english    # Both languages
python -m src.visualization --german              # German only
```  

## Pipeline Details

### Step 1: Data Cleaning

```bash
python -m src.cleaning
```

Preprocesses raw German complaint texts into two formats:
- **LDA**: Lemmatized, stopwords removed, lowercased
- **BERTopic**: Lightly cleaned, preserves structure

**Output**: `data/cleaned/cleaned_lda_berttopic.csv`

### Step 2: Vectorization (LDA Only)

```bash
python -m src.vectorization
```

Creates sparse matrices for LDA training:
- Bag-of-Words (BoW) matrix
- TF-IDF matrix
- Feature vocabularies

**Output**: `data/vectorized/lda/*.npz` and `*.joblib` files

### Step 3: Topic Modeling

```bash
python -m src.topic_modeling [--lda] [--bertopic]
```

Trains topic models with automatic optimization:
- **LDA**: Coherence-based K selection (BoW and TF-IDF variants)
- **BERTopic**: HDBSCAN clustering with multilingual embeddings

**Outputs**:
- Models: `data/topic_models/{lda,bertopic}/`
- Topics: `*_topics.csv`
- Metadata: `*_info.joblib`, `*_k_sweep.csv`

### Step 4: LLM Labeling (Optional)

```bash
python -m src.llm_topic_labeling [--lda] [--bertopic] [--german-only]
```

Generates human-readable topic names using GPT-4o-mini:
- Analyzes keywords + representative documents
- Creates German labels
- Translates to English (unless `--german-only`)

**Outputs**: `data/topic_models/*/german_labeled/` and `*/english_labeled/`

### Step 5: Visualization

```bash
python -m src.visualization [--german] [--english]
```

Creates comprehensive analysis outputs:
- CSV reports with topic distributions
- Interactive HTML visualizations (BERTopic)
- Word clouds and topic prevalence charts
- Representative document extracts
- Static plots are saved as PNG files, while interactive analyses (pyLDAvis and BERTopic visualizations) are exported as HTML files.

**Outputs**: `results/german_labeled/` and `results/english_labeled/`