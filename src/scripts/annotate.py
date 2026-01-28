"""Sentiment and Linguistic Alignment Annotation for Child-Adult Interactions."""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

warnings.filterwarnings("ignore")

# Import the alignment calculators (assuming they're in a separate file)
from pkg.annotator.alignment import LinguisticAlignmentSuite


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Annotate sentiment and linguistic alignment for child-adult interactions"
    )

    parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV/Excel file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save annotated results")
    parser.add_argument(
        "--child_column", type=str, default="utt_transcript_clean", help="Column name for child utterances"
    )
    parser.add_argument(
        "--adult_column", type=str, default="response_transcript_clean", help="Column name for adult responses"
    )
    parser.add_argument(
        "--file_column", type=str, default="transcript_file", help="Column name for grouping conversations"
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

    # Processing settings
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only first 100 rows")

    return parser.parse_args(argv)


class SentimentAnnotator:
    """Annotates adult turns with emotion-based sentiment scores.
    Optimized for child-adult educational interactions.
    """

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        """Initialize sentiment annotator.

        Args:
            model_name: HuggingFace model for emotion classification

        """
        print(f"Loading emotion model: {model_name}...")
        self.emotion_classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=-1,  # Use CPU; set to 0 for GPU
        )

        # Define emotion categories for child-adult interactions
        self.supportive_emotions = ["admiration", "approval", "caring", "optimism", "pride"]
        self.engagement_emotions = ["curiosity", "excitement", "amusement", "desire"]
        self.warmth_emotions = ["caring", "love", "gratitude", "joy"]
        self.negative_emotions = ["disappointment", "disapproval", "annoyance", "anger"]

    def _compute_composite_scores(self, emotion_dict: dict[str, float]) -> dict[str, float]:
        """Compute composite sentiment scores from emotion distribution."""
        supportiveness = np.mean([emotion_dict.get(e, 0) for e in self.supportive_emotions])

        engagement = np.mean([emotion_dict.get(e, 0) for e in self.engagement_emotions])

        warmth = np.mean([emotion_dict.get(e, 0) for e in self.warmth_emotions])

        negativity = np.mean([emotion_dict.get(e, 0) for e in self.negative_emotions])

        # Overall pedagogical positivity
        pedagogical_positivity = np.mean([supportiveness, engagement, warmth, 1 - negativity])

        return {
            "supportiveness": float(supportiveness),
            "engagement": float(engagement),
            "warmth": float(warmth),
            "negativity": float(negativity),
            "pedagogical_positivity": float(pedagogical_positivity),
            "approval": float(emotion_dict.get("approval", 0)),
            "curiosity": float(emotion_dict.get("curiosity", 0)),
            "caring": float(emotion_dict.get("caring", 0)),
        }

    def annotate_batch(self, texts: list[str]) -> pd.DataFrame:
        """Annotate a batch of texts with sentiment scores.

        Args:
            texts: List of adult utterances

        Returns:
            DataFrame with sentiment scores

        """
        results = []

        # Process in batches for efficiency
        for text in tqdm(texts, desc="Annotating sentiment"):
            if pd.isna(text) or text.strip() == "":
                # Handle empty/NaN texts
                results.append(
                    {
                        "supportiveness": 0.0,
                        "engagement": 0.0,
                        "warmth": 0.0,
                        "negativity": 0.0,
                        "pedagogical_positivity": 0.0,
                        "approval": 0.0,
                        "curiosity": 0.0,
                        "caring": 0.0,
                    }
                )
            else:
                emotions = self.emotion_classifier(text)[0]
                emotion_dict = {e["label"]: e["score"] for e in emotions}
                scores = self._compute_composite_scores(emotion_dict)
                results.append(scores)

        return pd.DataFrame(results)


def annotate_sentiment(df: pd.DataFrame, adult_column: str, emotion_model: str) -> pd.DataFrame:
    """Annotate adult turns with sentiment scores.

    Args:
        df: Input dataframe
        adult_column: Column name containing adult utterances
        emotion_model: HuggingFace model name

    Returns:
        DataFrame with sentiment score columns

    """
    print("\n" + "=" * 70)
    print("SENTIMENT ANNOTATION")
    print("=" * 70)

    annotator = SentimentAnnotator(model_name=emotion_model)

    # Extract adult utterances
    adult_texts = df[adult_column].tolist()

    # Annotate
    sentiment_scores = annotator.annotate_batch(adult_texts)

    # Add prefix to column names
    sentiment_scores.columns = ["sent_" + col for col in sentiment_scores.columns]

    return sentiment_scores


def annotate_alignment(
    df: pd.DataFrame,
    child_column: str,
    adult_column: str,
    spacy_model: str,
    semantic_model: str,
    exclude_stopwords: bool,
    file_column: str = None,
) -> pd.DataFrame:
    """Annotate child-adult pairs with linguistic alignment scores.

    Args:
        df: Input dataframe
        child_column: Column name containing child utterances
        adult_column: Column name containing adult responses
        spacy_model: spaCy model name
        semantic_model: SentenceTransformer model name
        exclude_stopwords: Whether to exclude stopwords
        file_column: Optional column for grouping conversations

    Returns:
        DataFrame with alignment score columns

    """
    print("\n" + "=" * 70)
    print("LINGUISTIC ALIGNMENT ANNOTATION")
    print("=" * 70)

    # Initialize alignment suite
    suite = LinguisticAlignmentSuite(
        spacy_model=spacy_model, semantic_model=semantic_model, exclude_stopwords=exclude_stopwords
    )

    # Extract utterances
    child_texts = df[child_column].tolist()
    adult_texts = df[adult_column].tolist()

    # Handle NaN values
    child_texts = ["" if pd.isna(text) else text for text in child_texts]
    adult_texts = ["" if pd.isna(text) else text for text in adult_texts]

    print(f"Processing {len(child_texts)} child-adult pairs...")

    # Compute alignments (batch processing for semantic alignment)
    alignment_results = suite.compute_batch(child_texts, adult_texts)

    # Create DataFrame
    alignment_df = pd.DataFrame(
        {
            "align_lexical": alignment_results["lexical_alignment"],
            "align_syntactic": alignment_results["syntactic_alignment"],
            "align_semantic": alignment_results["semantic_alignment"],
        }
    )

    return alignment_df


def main(argv):
    """Main function for sentiment and alignment annotation."""
    args = parse_args(argv)

    print("=" * 70)
    print("SENTIMENT & LINGUISTIC ALIGNMENT ANNOTATION")
    print("=" * 70)
    print(f"Input file: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Annotate sentiment: {args.annotate_sentiment}")
    print(f"Annotate alignment: {args.annotate_alignment}")

    # Load data
    print("\nLoading data...")
    if args.input_path.endswith(".csv"):
        df = pd.read_csv(args.input_path)
    elif args.input_path.endswith(".xlsx"):
        df = pd.read_excel(args.input_path)
    else:
        raise ValueError("Input file must be CSV or Excel format")

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Debug mode
    if args.debug:
        print("\n" + "!" * 70)
        print("DEBUG MODE: Processing only first 100 rows")
        print("!" * 70)
        df = df.head(100)

    # Verify required columns exist
    required_cols = [args.child_column, args.adult_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Initialize results list
    annotation_frames = [df.copy()]

    # Annotate sentiment
    if args.annotate_sentiment:
        sentiment_df = annotate_sentiment(df=df, adult_column=args.adult_column, emotion_model=args.emotion_model)
        annotation_frames.append(sentiment_df)
        print("✓ Sentiment annotation complete")
        print(f"  Added columns: {sentiment_df.columns.tolist()}")

    # Annotate linguistic alignment
    if args.annotate_alignment:
        alignment_df = annotate_alignment(
            df=df,
            child_column=args.child_column,
            adult_column=args.adult_column,
            spacy_model=args.spacy_model,
            semantic_model=args.semantic_model,
            exclude_stopwords=args.exclude_stopwords,
            file_column=args.file_column,
        )
        annotation_frames.append(alignment_df)
        print("✓ Alignment annotation complete")
        print(f"  Added columns: {alignment_df.columns.tolist()}")

    # Combine all annotations
    print("\nCombining annotations...")
    result_df = pd.concat(annotation_frames, axis=1)

    # Remove duplicate columns (if any)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename
    input_filename = os.path.basename(args.input_path)
    base_name = os.path.splitext(input_filename)[0]

    if args.debug:
        output_filename = f"{base_name}_annotated_debug.csv"
    else:
        output_filename = f"{base_name}_annotated.csv"

    output_path = os.path.join(args.output_dir, output_filename)

    # Save
    if output_path.endswith(".csv"):
        result_df.to_csv(output_path, index=False)
    elif output_path.endswith(".xlsx"):
        result_df.to_excel(output_path, index=False)

    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE")
    print("=" * 70)
    print(f"Output saved to: {output_path}")
    print(f"Total rows: {len(result_df)}")
    print(f"Total columns: {len(result_df.columns)}")

    # Display sample results
    print("\nSample of annotated data:")
    display_cols = [args.child_column, args.adult_column]
    if args.annotate_sentiment:
        display_cols.extend(["sent_supportiveness", "sent_pedagogical_positivity"])
    if args.annotate_alignment:
        display_cols.extend(["align_lexical", "align_syntactic", "align_semantic"])

    available_cols = [col for col in display_cols if col in result_df.columns]
    print(result_df[available_cols].head(3).to_string())

    # Summary statistics
    if args.annotate_sentiment:
        print("\nSentiment Score Statistics:")
        sentiment_cols = [col for col in result_df.columns if col.startswith("sent_")]
        print(result_df[sentiment_cols].describe())

    if args.annotate_alignment:
        print("\nAlignment Score Statistics:")
        alignment_cols = [col for col in result_df.columns if col.startswith("align_")]
        print(result_df[alignment_cols].describe())


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
