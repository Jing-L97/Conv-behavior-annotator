"""Sentiment and Linguistic Alignment Annotation for Child-Adult Interactions."""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
from pkg import settings
from pkg.annotator.alignment import LinguisticAlignmentSuite
from pkg.annotator.sentiment import SentimentAnnotator
from pkg.settings import get_torch_device

##############################
# Argument Parsing
##############################


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Annotate sentiment and linguistic alignment for child-adult interactions"
    )

    parser.add_argument(
        "--input_path", type=str, default="raw/conversations_min_age_10.csv", help="Path to input CSV/Excel file"
    )
    parser.add_argument("--output_dir", type=str, default="annotated/", help="Directory to save annotated results")
    parser.add_argument(
        "--child_column", type=str, default="utt_transcript_clean", help="Column name for child utterances"
    )
    parser.add_argument(
        "--adult_column", type=str, default="response_transcript_clean", help="Column name for adult responses"
    )

    # Sentiment analysis settings
    parser.add_argument(
        "--emotion_model",
        type=str,
        default="SamLowe/roberta-base-go_emotions",
        help="HuggingFace model for emotion classification",
    )
    parser.add_argument("--annotate_sentiment", action="store_true", help="Whether to annotate sentiment scores")

    # Linguistic alignment settings
    parser.add_argument("--annotate_alignment", action="store_true", help="Whether to annotate linguistic alignment")
    parser.add_argument(
        "--spacy_model", type=str, default="en_core_web_sm", help="spaCy model for lexical/syntactic analysis"
    )
    parser.add_argument(
        "--semantic_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for semantic alignment",
    )
    parser.add_argument("--exclude_stopwords", action="store_true", help="Exclude stopwords in lexical alignment")
    parser.add_argument(
        "--exclude_interjections", action="store_true", help="Exclude interjections in lexical alignment"
    )
    # Processing settings
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only first 100 rows")

    return parser.parse_args(argv)


##############################
# Annotate Functions
##############################


def annotate_sentiment(
    df: pd.DataFrame, adult_column: str, emotion_model: str, device: str | None, batch_size: int
) -> pd.DataFrame:
    """Annotate adult turns with sentiment scores."""
    print("\n" + "=" * 70)
    print("SENTIMENT ANNOTATION")
    print("=" * 70)

    annotator = SentimentAnnotator(model_name=emotion_model, device=device)

    # Extract adult utterances
    adult_texts = df[adult_column].tolist()

    # Annotate
    sentiment_scores = annotator.annotate_batch(adult_texts, batch_size)

    # Add prefix to column names
    sentiment_scores.columns = ["sent_" + col for col in sentiment_scores.columns]

    return sentiment_scores


def annotate_alignment(
    df: pd.DataFrame,
    child_column: str,
    adult_column: str,
    spacy_model: str = "en_core_web_sm",
    semantic_model: str = "all-MiniLM-L6-v2",
    exclude_stopwords: bool = False,
    exclude_interjections: bool = False,
    device: str | None = None,
) -> pd.DataFrame:
    """Annotate child-adult pairs with linguistic alignment scores."""
    print("\n" + "=" * 70)
    print("LINGUISTIC ALIGNMENT ANNOTATION")
    print("=" * 70)

    # Initialize alignment suite
    suite = LinguisticAlignmentSuite(
        spacy_model=spacy_model,
        semantic_model=semantic_model,
        exclude_stopwords=exclude_stopwords,
        exclude_interjections=exclude_interjections,
        device=device,
    )

    # Extract utterances
    child_texts = df[child_column].fillna("").tolist()
    adult_texts = df[adult_column].fillna("").tolist()

    print(f"Processing {len(child_texts)} child-adult pairs...")

    # Compute batch alignments
    alignment_results = suite.compute_batch(child_texts, adult_texts)

    # Build DataFrame with separate lexical scores
    alignment_df = pd.DataFrame(
        {
            "align_lexical_unigram": alignment_results["lexical_unigram_alignment"],
            "align_lexical_bigram": alignment_results["lexical_bigram_alignment"],
            "align_syntactic": alignment_results["syntactic_alignment"],
            # "align_semantic": alignment_results["semantic_alignment"],
        }
    )

    return alignment_df


##############################
# main function
##############################


def main(argv):
    """Main function for sentiment and alignment annotation."""
    args = parse_args(argv)
    device = get_torch_device()

    input_path: Path = settings.PATH.dataset_root / args.input_path
    output_dir: Path = settings.PATH.dataset_root / args.output_dir

    print("=" * 70)
    print("SENTIMENT & LINGUISTIC ALIGNMENT ANNOTATION")
    print("=" * 70)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Annotate sentiment: {args.annotate_sentiment}")
    print(f"Annotate alignment: {args.annotate_alignment}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")

    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Input file must be CSV or Excel format")

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # ------------------------------------------------------------------
    # Debug mode
    # ------------------------------------------------------------------
    if args.debug:
        print("\n" + "!" * 70)
        print("DEBUG MODE: Processing only first 100 rows")
        print("!" * 70)
        df = df.head(100)

    # ------------------------------------------------------------------
    # Verify required columns
    # ------------------------------------------------------------------
    required_cols = [args.child_column, args.adult_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------
    annotation_frames = [df.copy()]

    if args.annotate_sentiment:
        sentiment_df = annotate_sentiment(
            df=df,
            adult_column=args.adult_column,
            emotion_model=args.emotion_model,
            device=device,
            batch_size=args.batch_size,  # optional
        )
        annotation_frames.append(sentiment_df)
        print("✓ Sentiment annotation complete")
        print(f"  Added columns: {sentiment_df.columns.tolist()}")

    if args.annotate_alignment:
        alignment_df = annotate_alignment(
            df=df,
            child_column=args.child_column,
            adult_column=args.adult_column,
            spacy_model=args.spacy_model,
            semantic_model=args.semantic_model,
            exclude_stopwords=args.exclude_stopwords,
            exclude_interjections=args.exclude_interjections,
            device=device,
        )
        annotation_frames.append(alignment_df)
        print("✓ Alignment annotation complete")
        print(f"  Added columns: {alignment_df.columns.tolist()}")

    # ------------------------------------------------------------------
    # Combine results
    # ------------------------------------------------------------------
    print("\nCombining annotations...")
    result_df = pd.concat(annotation_frames, axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ""
    if args.exclude_stopwords:
        suffix += "_no_stopwords"
    if args.exclude_interjections:
        suffix += "_no_intjs"

    out_file_name = f"{input_path.stem}{suffix}_debug.csv" if args.debug else f"{input_path.stem}{suffix}.csv"

    output_path: Path = output_dir / out_file_name

    if output_path.suffix == ".csv":
        result_df.to_csv(output_path, index=False)
    elif output_path.suffix in {".xlsx", ".xls"}:
        result_df.to_excel(output_path, index=False)

    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE")
    print("=" * 70)
    print(f"Output saved to: {output_path}")
    print(f"Total rows: {len(result_df)}")
    print(f"Total columns: {len(result_df.columns)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
