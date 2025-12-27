"""
main.py
-------

Orchestrating pipeline for Munich complaints topic modeling analysis.

This script coordinates the execution of all topic modeling pipeline steps,
from raw data cleaning through model training to visualization generation.
Supports flexible execution of individual steps or full end-to-end runs.

Pipeline Steps
--------------
1. Data cleaning (cleaning.py) - Preprocesses raw German complaint texts
2. Vectorization (vectorization.py) - Creates BoW and TF-IDF matrices for LDA
3. Topic modeling (topic_modeling.py) - Trains LDA and/or BERTopic models
4. LLM labeling (llm_topic_labeling.py) - Generates human-readable topic names (optional)
5. Visualization (visualization.py) - Creates interactive plots and analysis outputs


Outputs
-------
Coordinates creation of:
- Cleaned data → data/cleaned/
- Vectorized matrices → data/vectorized/lda/
- Trained models → data/topic_models/
- Topic labels → data/topic_models/*/german_labeled/ and */english_labeled/
- Visualizations → results/ and language-specific subdirectories


How to Run
----------
1) Full pipeline (both models, no LLM labels):

    python -m src.main --clean --vectorize --train --visualize

2) Full pipeline (both models, bilingual LLM labels):

    python -m src.main --clean --vectorize --train --label --visualize --german --english

3) LDA-only pipeline (no LLM labels):

    python -m src.main --clean --vectorize --train --models lda --visualize

4) LDA-only pipeline (with bilingual LLM labels):

    python -m src.main --clean --vectorize --train --models lda --label --visualize --german --english

5) BERTopic-only pipeline (no LLM labels):

    python -m src.main --clean --train --models bertopic --visualize

6) BERTopic-only pipeline (with bilingual LLM labels):

    python -m src.main --clean --train --models bertopic --label --visualize --german --english


Command-Line Arguments
----------------------
Pipeline steps (flags):
  --clean              Run data cleaning pipeline
  --vectorize          Run vectorization (BoW + TF-IDF for LDA)
  --train              Train topic models
  --label              Generate LLM-based topic labels (requires API key)
  --visualize          Create visualizations and analysis outputs

Model selection:
  --models {lda,bertopic,all}
                       Which models to train/label (default: all)

Language options:
  --german             Generate/use German topic labels
  --english            Generate/use English topic labels


Prerequisites
-------------
- Raw complaint data in data/raw/
- OpenAI API key in .env file (required for --label step)
- All dependencies from requirements.txt installed


Notes
-----
- Steps must be run in order for first-time execution
- Vectorization is only needed for LDA; BERTopic uses raw cleaned text
- LLM labeling is optional but recommended for interpretable results
- German labels are generated first; English labels are translations
- Individual steps can be re-run if previous outputs exist
- API costs for LLM labeling: ~$0.20-0.40 for full bilingual pipeline
"""

import argparse
import subprocess
import sys

# Import config to check environment
from src.config import OPENAI_API_KEY


def run_cleaning():
    """
    Executes the data cleaning pipeline for Munich complaints.

    Runs cleaning.py as a subprocess to clean raw complaint text and save
    outputs to data/cleaned/. Creates both LDA-optimized and BERTopic-optimized
    text columns.

    Notes
    -----
    - Input: data/raw/munich_open311_2020-01-01_to_2025-12-01.csv
    - Output: data/cleaned/cleaned_lda_berttopic.csv
    - Exits with code 1 if cleaning fails
    """
    print("\n" + "="*70)
    print("STEP 1: DATA CLEANING")
    print("="*70)
    print("Running cleaning pipeline for raw Munich complaints...")
    print("Input:  data/raw/munich_open311_2020-01-01_to_2025-12-01.csv")
    print("Output: data/cleaned/cleaned_lda_berttopic.csv")
    print()

    cmd = [sys.executable, "-m", "src.cleaning"]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Cleaning step failed!")
        sys.exit(1)
    else:
        print("\n✅ Cleaning completed successfully!")


def run_vectorization():
    """
    Executes the vectorization pipeline to create BoW and TF-IDF matrices for LDA.

    Runs vectorization.py as a subprocess to create sparse matrices and vocabularies
    in data/vectorized/lda/. This step is required for LDA training but not used
    by BERTopic.

    Notes
    -----
    - Input: data/cleaned/cleaned_lda_berttopic.csv (lda_description column)
    - Output: data/vectorized/lda/*.npz, *.joblib
    - Exits with code 1 if vectorization fails
    """
    print("\n" + "="*70)
    print("STEP 2: VECTORIZATION")
    print("="*70)
    print("Creating BoW and TF-IDF matrices for LDA...")
    print("Input:  data/cleaned/cleaned_lda_berttopic.csv (lda_description column)")
    print("Output: data/vectorized/lda/*.npz, *.joblib")
    print()

    cmd = [sys.executable, "-m", "src.vectorization"]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Vectorization step failed!")
        sys.exit(1)
    else:
        print("\n✅ Vectorization completed successfully!")


def run_training(model_type: str):
    """
    Executes topic model training (LDA and/or BERTopic).

    Runs topic_modeling.py as a subprocess to train models and save artifacts
    to data/topic_models/. LDA uses vectorized matrices; BERTopic uses raw
    cleaned text.

    Parameters
    ----------
    model_type : str
        Model(s) to train. Must be one of:
        - "lda": Train LDA with BoW and TF-IDF
        - "bertopic": Train BERTopic only
        - "all": Train both LDA and BERTopic (default)

    Notes
    -----
    - Output: data/topic_models/lda/ and/or data/topic_models/bertopic/
    - Exits with code 1 if training fails

    Raises
    ------
    ValueError
        If model_type is not one of {"lda", "bertopic", "all"}
    """
    print("\n" + "="*70)
    print("STEP 3: TOPIC MODELING")
    print("="*70)

    if model_type == "all":
        print("Training both LDA (BoW + TF-IDF) and BERTopic...")
        cmd_args = []
    elif model_type == "lda":
        print("Training LDA models (BoW + TF-IDF)...")
        cmd_args = ["--lda"]
    elif model_type == "bertopic":
        print("Training BERTopic model...")
        cmd_args = ["--bertopic"]
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    print("Output: data/topic_models/lda/ and/or data/topic_models/bertopic/")
    print()

    cmd = [sys.executable, "-m", "src.topic_modeling"] + cmd_args
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Training step failed!")
        sys.exit(1)
    else:
        print("\n✅ Training completed successfully!")


def run_labeling(model_type: str, german: bool, english: bool):
    """
    Executes LLM-based topic labeling using OpenAI GPT.

    Runs llm_topic_labeling.py as a subprocess to generate human-readable
    topic names and save to language-specific directories. Requires OpenAI
    API key in environment.

    Parameters
    ----------
    model_type : str
        Model(s) to label. Must be one of:
        - "lda": Label LDA topics only
        - "bertopic": Label BERTopic topics only
        - "all": Label both models (default)
    german : bool
        If True, generate German topic labels
    english : bool
        If True, generate English topic labels (translations of German)

    Notes
    -----
    - Requires: OPENAI_API_KEY environment variable (in .env file)
    - Output: data/topic_models/*/german_labeled/ and/or */english_labeled/
    - Model: gpt-4o-mini
    - Estimated cost: $0.10-0.20 per language
    - English labels are translations of German labels
    - If neither german nor english is True, defaults to generating both
    - Exits with code 1 if API key is missing or labeling fails
    """
    print("\n" + "="*70)
    print("STEP 4: LLM TOPIC LABELING")
    print("="*70)

    # Validate API key
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment!")
        print("   Please add your API key to the .env file.")
        print("   See .env.example for template.")
        sys.exit(1)

    # Build command arguments
    cmd_args = []

    # Model selection
    if model_type == "lda":
        cmd_args.append("--lda")
        print("Labeling LDA topics...")
    elif model_type == "bertopic":
        cmd_args.append("--bertopic")
        print("Labeling BERTopic topics...")
    else:  # "all"
        print("Labeling both LDA and BERTopic topics...")

    # Language selection
    if german and not english:
        cmd_args.append("--german-only")
        print("Language: German only")
    elif german and english:
        print("Language: German + English (bilingual)")
        # Default behavior: generates both
    elif english and not german:
        print("⚠️  Warning: English labels require German labels first (translation).")
        print("   Generating both German and English labels...")
        # Default behavior handles this

    if not german and not english:
        print("⚠️  Warning: No language specified. Defaulting to German + English.")

    print("Output: data/topic_models/*/german_labeled/ and/or */english_labeled/")
    print("Using LLM: gpt-4o-mini")
    print("Estimated cost: $0.10-0.20 per language")
    print()

    cmd = [sys.executable, "-m", "src.llm_topic_labeling"] + cmd_args
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Labeling step failed!")
        sys.exit(1)
    else:
        print("\n✅ LLM labeling completed successfully!")


def run_visualization(german: bool, english: bool):
    """
    Executes visualization and analysis output generation.

    Runs visualization.py as a subprocess to generate interactive plots,
    charts, and HTML visualizations. Saves to results/ (standard) and
    language-specific subdirectories (labeled).

    Parameters
    ----------
    german : bool
        If True, create German visualizations (requires German labels)
    english : bool
        If True, create English visualizations (requires English labels)

    Notes
    -----
    - Output: results/ (standard keyword-based visualizations)
    - Output: results/german_labeled/ (German LLM-labeled visualizations)
    - Output: results/english_labeled/ (English LLM-labeled visualizations)
    - If neither german nor english is True, generates standard visualizations
    - Exits with code 1 if visualization fails
    """
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)

    # Build command arguments
    cmd_args = []

    if german:
        cmd_args.append("--german")
        print("Generating German visualizations...")

    if english:
        cmd_args.append("--english")
        print("Generating English visualizations...")

    if not german and not english:
        print("Generating standard visualizations (keyword-based topic names)...")

    print("Output: results/ (standard), results/german_labeled/, results/english_labeled/")
    print()

    cmd = [sys.executable, "-m", "src.visualization"] + cmd_args
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Visualization step failed!")
        sys.exit(1)
    else:
        print("\n✅ Visualization completed successfully!")


def print_pipeline_summary(args):
    """
    Prints a summary of the pipeline configuration before execution.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing pipeline configuration
    """
    print("\n" + "="*70)
    print("MUNICH COMPLAINTS TOPIC MODELING PIPELINE")
    print("="*70)
    print("\nPipeline Configuration:")
    print(f"  Models:             {args.models}")
    print(f"  German labels:      {'Yes' if args.german else 'No'}")
    print(f"  English labels:     {'Yes' if args.english else 'No'}")
    print("\nSteps to execute:")

    steps = []
    if args.clean:
        steps.append("  1. Data cleaning")
    if args.vectorize:
        steps.append("  2. Vectorization (BoW + TF-IDF)")
    if args.train:
        steps.append(f"  3. Topic modeling ({args.models})")
    if args.label:
        steps.append("  4. LLM topic labeling")
    if args.visualize:
        steps.append("  5. Visualization generation")

    if steps:
        print("\n".join(steps))
    else:
        print("  (No steps selected - use flags like --clean, --train, etc.)")

    print("="*70 + "\n")


def main():
    """
    Entry point for the topic modeling pipeline orchestrator.

    Parses command-line arguments and executes the requested pipeline steps
    in the correct order: clean → vectorize → train → label → visualize.

    Each step is optional and can be run independently (assuming prerequisites exist).

    Notes
    -----
    - Steps must be run in order for first-time execution
    - Individual steps can be re-run if previous outputs exist
    - Use --help flag for full usage documentation and examples
    """
    parser = argparse.ArgumentParser(
        description="Run Munich complaints topic modeling pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (both models, no LLM labels)
  python -m src.main --clean --vectorize --train --visualize

  # Full pipeline (both models, bilingual LLM labels)
  python -m src.main --clean --vectorize --train --label --visualize --german --english

  # LDA-only pipeline (no LLM labels)
  python -m src.main --clean --vectorize --train --models lda --visualize

  # LDA-only pipeline (with LLM labels)
  python -m src.main --clean --vectorize --train --models lda --label --visualize --german

  # BERTopic-only pipeline (no LLM labels)
  python -m src.main --clean --train --models bertopic --visualize

  # BERTopic-only pipeline (with LLM labels)
  python -m src.main --clean --train --models bertopic --label --visualize --german
        """
    )

    # Pipeline steps
    parser.add_argument("--clean", action="store_true",
                        help="Run data cleaning pipeline")
    parser.add_argument("--vectorize", action="store_true",
                        help="Run vectorization (BoW + TF-IDF for LDA)")
    parser.add_argument("--train", action="store_true",
                        help="Train topic models")
    parser.add_argument("--label", action="store_true",
                        help="Generate LLM-based topic labels (requires API key)")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualizations and analysis outputs")

    # Model selection
    parser.add_argument("--models", choices=["lda", "bertopic", "all"], default="all",
                        help="Which models to train/label (default: all)")

    # Language options
    parser.add_argument("--german", action="store_true",
                        help="Generate/use German topic labels")
    parser.add_argument("--english", action="store_true",
                        help="Generate/use English topic labels")

    args = parser.parse_args()

    # Print summary
    print_pipeline_summary(args)

    # Check if any steps are selected
    if not any([args.clean, args.vectorize, args.train, args.label, args.visualize]):
        print("⚠️  No pipeline steps selected!")
        print("   Use flags like --clean, --train, --visualize to specify steps.")
        print("   Run with --help for usage examples.")
        sys.exit(0)

    # Execute pipeline steps in order
    try:
        if args.clean:
            run_cleaning()

        if args.vectorize:
            run_vectorization()

        if args.train:
            run_training(args.models)

        if args.label:
            run_labeling(args.models, args.german, args.english)

        if args.visualize:
            run_visualization(args.german, args.english)

        # Final success message
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)

        # Print output locations
        print("\nOutput locations:")
        if args.clean:
            print("  Cleaned data:       data/cleaned/")
        if args.vectorize:
            print("  Vectorized data:    data/vectorized/lda/")
        if args.train:
            print("  Trained models:     data/topic_models/")
        if args.label:
            if args.german:
                print("  German labels:      data/topic_models/*/german_labeled/")
            if args.english:
                print("  English labels:     data/topic_models/*/english_labeled/")
        if args.visualize:
            print("  Visualizations:     results/")
            if args.german:
                print("                      results/german_labeled/")
            if args.english:
                print("                      results/english_labeled/")

        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
