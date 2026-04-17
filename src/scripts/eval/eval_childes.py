import argparse
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from pkg.rlhf.eval.fast_eval import (
    build_feature_list,
    compute_entropy_reg,
    compute_feature_metrics_for_utts,
)
from pkg.rlhf.eval.gen_util import FeatureExtractor
from pkg.rlhf.eval.grammar_util import (
    compute_scores_childes_grammaticality,
    compute_scores_gec,
    load_childes_grammar_model,
    load_gec_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Read utterances from CSV
# ----------------------------
def read_utterances_from_csv(csv_path: str, utt_column: str) -> list[str]:
    """Read utterances from a CSV file, one utterance per row.

    Args:
        csv_path:   Path to the input CSV file.
        utt_column: Name of the column containing the utterances.

    Returns:
        A list of non-empty utterance strings.

    """
    df = pd.read_csv(csv_path)

    if utt_column not in df.columns:
        raise ValueError(f"Column '{utt_column}' not found in '{csv_path}'. Available columns: {list(df.columns)}")

    utterances = df[utt_column].dropna().astype(str).str.strip().loc[lambda s: s != ""].tolist()

    print(f"[Read] {len(utterances)} valid utterances from '{csv_path}' (column='{utt_column}')")
    return utterances


# ----------------------------
# Grammaticality scoring in batches
# ----------------------------
def score_in_batches(utterances: list[str], score_fn, model, tokenizer, eval_batch_size: int) -> list:
    """Run a grammaticality scoring function in batches of eval_batch_size."""
    scores = []
    for start in range(0, len(utterances), eval_batch_size):
        batch = utterances[start : start + eval_batch_size]
        batch_scores = score_fn(batch, model, tokenizer)
        scores.extend(list(batch_scores))
    return scores


# ----------------------------
# Core evaluation (no generation)
# ----------------------------
def eval_grammaticality_and_features(
    utterances: list[str],
    childes_grammar_model,
    childes_grammar_model_tokenizer,
    gec_model,
    gec_model_tokenizer,
    feature_extractor,
    sent_model,
    feature_list,
    output_utts_csv: str,
    eval_batch_size: int = 1024,
) -> dict:
    """Evaluate a flat list of utterances for grammaticality and linguistic features."""
    if not utterances:
        print("[Warning] No utterances to evaluate.")
        return {}

    n = len(utterances)
    print(f"Scoring grammaticality on {n} utterances ...")

    # --- Grammaticality ---
    all_scores_childes = score_in_batches(
        utterances,
        compute_scores_childes_grammaticality,
        childes_grammar_model,
        childes_grammar_model_tokenizer,
        eval_batch_size,
    )
    all_scores_gec = score_in_batches(
        utterances,
        compute_scores_gec,
        gec_model,
        gec_model_tokenizer,
        eval_batch_size,
    )

    # --- Linguistic feature metrics ---
    turn_df, feat_agg = compute_feature_metrics_for_utts(utterances, feature_extractor, sent_model, feature_list)

    # --- Build per-utterance DataFrame ---
    utts_df = pd.DataFrame(
        {
            "utterance": utterances,
            "grammaticality_childes_score": all_scores_childes[:n],
            "grammaticality_gec_score": all_scores_gec[:n],
        }
    )

    if not turn_df.empty:
        if len(turn_df) == n:
            turn_df = turn_df.reset_index(drop=True)
            utts_df = pd.concat([utts_df, turn_df], axis=1)
        else:
            print(
                f"[Warning] turn_df has {len(turn_df)} rows but expected {n}; skipping per-utterance feature merge.",
                flush=True,
            )

    # --- Save per-utterance CSV ---
    os.makedirs(os.path.dirname(os.path.abspath(output_utts_csv)), exist_ok=True)
    utts_df.to_csv(output_utts_csv, index=False)
    print(f"[Saved] per-utterance results -> {output_utts_csv}", flush=True)

    # --- Aggregate summary ---
    results = {
        "grammaticality_childes": float(np.mean(all_scores_childes)) if all_scores_childes else np.nan,
        "grammaticality_gec": float(np.mean(all_scores_gec)) if all_scores_gec else np.nan,
        "num_scored_sentences": n,
        "entropy_reg": compute_entropy_reg(utterances),
    }
    results.update(feat_agg)

    print(
        f"childes={results['grammaticality_childes']:.3f} | "
        f"gec={results['grammaticality_gec']:.3f} | "
        f"scored={results['num_scored_sentences']}"
    )
    return results


# ----------------------------
# Entry point
# ----------------------------
def eval_csv(args):
    # Load evaluation models
    childes_grammar_model, childes_grammar_model_tokenizer = load_childes_grammar_model(args.eval_model_path)
    gec_model, gec_model_tokenizer = load_gec_model()

    # Load feature resources
    word_info = pd.read_csv(args.word_info_path)
    func_info = pd.read_csv(args.func_info_path)
    sent_model = SentenceTransformer(args.sent_model_path)
    feature_extractor = FeatureExtractor(word_info, func_info, embedding_model=sent_model)
    feature_list = build_feature_list(args.fea_set)

    utterances = read_utterances_from_csv(args.input_csv, args.utt_column)

    if args.debug:
        utterances = utterances[:50]
        print(f"[Debug] Truncated to {len(utterances)} utterances.")

    results = eval_grammaticality_and_features(
        utterances=utterances,
        childes_grammar_model=childes_grammar_model,
        childes_grammar_model_tokenizer=childes_grammar_model_tokenizer,
        gec_model=gec_model,
        gec_model_tokenizer=gec_model_tokenizer,
        feature_extractor=feature_extractor,
        sent_model=sent_model,
        feature_list=feature_list,
        output_utts_csv=args.output_utts_csv,
        eval_batch_size=args.eval_batch_size,
    )

    # --- Save aggregate results ---
    df = pd.DataFrame([results])
    output_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    cols_to_show = [
        "grammaticality_childes",
        "grammaticality_gec",
        "conc_mean",
        "lex_den_mean",
        "func_den_new_mean",
        "tree_depth_mean",
        "clause_mean",
        "sem_ent_set",
        "sem_div_set",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    if cols_to_show:
        print("\nSummary:\n", df[cols_to_show].to_string(index=False))

    print(f"\nSaved results to: {args.output_csv}")


# ----------------------------
# CLI
# ----------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Evaluate utterances read from a CSV using the grammaticality and feature pipeline."
    )

    # --- Input ---
    p.add_argument("--input_csv", type=str, required=True, help="CSV file containing utterances to evaluate.")
    p.add_argument(
        "--utt_column",
        type=str,
        default="utterance",
        help="Name of the column holding the utterance text (default: 'utterance').",
    )

    # --- Evaluation models ---
    p.add_argument("--eval_model_path", type=str, required=True)
    p.add_argument(
        "--eval_batch_size", type=int, default=1024, help="Number of utterances per batch when scoring grammaticality."
    )

    # --- Feature resources ---
    p.add_argument("--word_info_path", type=str, required=True)
    p.add_argument("--func_info_path", type=str, required=True)
    p.add_argument("--sent_model_path", type=str, default="paraphrase-MiniLM-L6-v2")
    p.add_argument(
        "--fea_set",
        type=str,
        nargs="+",
        default=["word", "syn", "div", "semEnt"],
        help="Feature groups to compute: word syn div semEnt",
    )

    # --- Outputs ---
    p.add_argument(
        "--output_utts_csv", type=str, default="utterances_eval.csv", help="Path for the per-utterance results CSV."
    )
    p.add_argument("--output_csv", type=str, default="results.csv", help="Path for the aggregate results CSV.")

    # --- Debug ---
    p.add_argument("--debug", action="store_true", help="If set, evaluate only the first 50 utterances.")

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_csv(args)
