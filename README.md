# Munich Complaints Analysis

Topic modeling and analysis of Munich city complaints using LDA and BERTopic with LLM-generated topic labels.

## Features

- **Dual Topic Modeling**: LDA (Bag-of-Words & TF-IDF) and BERTopic with transformer embeddings
- **LLM-Generated Topic Labels**: Human-readable topic names using GPT-4o-mini
- **Bilingual Support**: Generate topic labels in both German (original) and English
- **Interactive Visualizations**: pyLDAvis for LDA, interactive HTML for BERTopic
- **Comprehensive Analysis**: Topic distributions, word clouds, and representative documents

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd complaints_analysis
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
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

### 5. Verify setup
```bash
# Check that your API key is loaded
python -c "from src.config import OPENAI_API_KEY; print('API key configured!' if OPENAI_API_KEY else 'API key missing')"
```

## Usage

### Basic Topic Modeling (without LLM labels)
```bash
# Run LDA and BERTopic with standard keyword-based topic names
python -m src.topic_modeling
python -m src.visualization
```

### LLM-Enhanced Topic Modeling (German only)
```bash
# Generate German topic labels using GPT-4o-mini
python -m src.topic_modeling --use-llm
python -m src.visualization --german
```

### Full Bilingual Pipeline (German + English)
```bash
# Generate both German and English topic labels
python -m src.topic_modeling --use-llm --english
python -m src.visualization --german --english
```

**Note**: Run commands sequentially (topic modeling first, then visualization). The visualization step depends on artifacts created by topic modeling.

## Project Structure

```
complaints_analysis/
├── data/
│   ├── raw/                          # Raw complaint data
│   ├── cleaned/                      # Cleaned datasets
│   ├── vectorized/                   # Vectorized data for LDA
│   └── topic_models/
│       ├── lda/
│       │   ├── german/              # LDA German labeled artifacts
│       │   └── english/             # LDA English labeled artifacts
│       └── bertopic/
│           ├── german/              # BERTopic German labeled artifacts
│           └── english/             # BERTopic English labeled artifacts
├── results/
│   ├── german/                      # German visualizations
│   └── english/                     # English visualizations
├── src/
│   ├── config.py                    # Configuration and paths
│   ├── cleaning.py                  # Data cleaning
│   ├── topic_modeling.py            # LDA and BERTopic training
│   ├── topic_labeling.py            # LLM-based topic naming
│   ├── visualization.py             # Visualization generation
│   └── data_loader.py               # Data loading utilities
└── notebooks/                       # Jupyter notebooks for exploration
```

## Output Files

### Without LLM Labels (main directories)
- `results/00_lda_bow_topics.csv` - LDA topic-word distributions
- `results/06_bertopic_all_topics.csv` - BERTopic topic information
- And other standard outputs (files 00-04, 06, 08)

### With LLM Labels (language subdirectories)
- `results/german/05_lda_bow_topics_labeled.csv` - German topic labels
- `results/german/07_bertopic_all_topics_labeled.csv` - German BERTopic labels
- `results/english/05_lda_bow_topics_labeled_en.csv` - English topic labels
- `results/english/07_bertopic_all_topics_labeled_en.csv` - English BERTopic labels
- Interactive visualizations, word clouds, and analysis files (files 09-14)

## Requirements

- Python 3.8+
- OpenAI API key (for LLM-based topic labeling)
- See `requirements.txt` for full dependency list

## License

[Your License Here]