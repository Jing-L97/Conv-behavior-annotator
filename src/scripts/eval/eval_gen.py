import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pkg.rlhf.utilities import DEFAULT_MAX_GENERATION_LEN, DEFAULT_MIN_GENERATION_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Generation
# ----------------------------
def generate(model, tokenizer, batch_size, output_max_length):
    generation_kwargs = dict(
        min_length=-1,
        max_new_tokens=output_max_length,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    bos_tensor = torch.full((batch_size, 1), tokenizer.bos_token_id, device=device)

    with torch.no_grad():
        utts = model.generate(bos_tensor, **generation_kwargs)

    utts_decoded = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in utts]
    return {"utts": utts, "utts_decoded": utts_decoded}


def filter_utts_for_scoring(batch, tokenizer):
    utterances = batch["utts_decoded"]
    utt_lengths = [(utt != torch.tensor(tokenizer.pad_token_id)).sum().item() - 1 for utt in batch["utts"]]

    filtered = []
    for utt, length, utt_tokens in zip(utterances, utt_lengths, batch["utts"], strict=False):
        if length <= DEFAULT_MIN_GENERATION_LEN:
            continue
        if tokenizer.eos_token_id not in utt_tokens:
            continue
        filtered.append((utt, length))

    if not filtered:
        return [], [], []

    utterances_f, lengths_f = zip(*filtered, strict=False)
    return list(utterances_f), list(lengths_f), filtered


# ----------------------------
# Grammaticality scoring with eval_batch_size
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
# Combined evaluation loop
# ----------------------------
def eval_grammaticality_and_features(
    model,
    tokenizer,
    childes_grammar_model,
    childes_grammar_model_tokenizer,
    gec_model,
    gec_model_tokenizer,
    feature_extractor,
    sent_model,
    feature_list,
    model_path,
    output_utts_csv,
    num_batches=200,
    batch_size=50,
    output_max_length=DEFAULT_MAX_GENERATION_LEN,
    eval_batch_size=1024,
):
    from pkg.rlhf.eval.fast_eval import (
        compute_entropy_reg,
        compute_feature_metrics_for_utts,
    )
    from pkg.rlhf.eval.grammar_util import compute_scores_childes_grammaticality, compute_scores_gec

    all_scores_childes = []
    all_scores_gec = []
    all_lengths = []
    all_utts = []
    total_generated = 0

    for i in range(num_batches):
        last_ping = time.time()

        if time.time() - last_ping > 300:
            print(f"[Heartbeat] job still running at batch {i}", flush=True)
            last_ping = time.time()

        if i % 5 == 0:
            print(
                f"[{model_path}] Batch {i}/{num_batches} | "
                f"generated so far: {total_generated} | "
                f"scored so far: {len(all_utts)}",
                flush=True,
            )

        batch = generate(model, tokenizer, batch_size, output_max_length)
        print(f"  generated {len(batch['utts'])} utterances", flush=True)
        utterances, lengths, _ = filter_utts_for_scoring(batch, tokenizer)

        total_generated += len(batch["utts"])
        if not utterances:
            continue

        # Grammaticality scoring in eval_batch_size chunks
        print(f"  scoring grammaticality on {len(utterances)} utterances", flush=True)
        s_childes = score_in_batches(
            utterances,
            compute_scores_childes_grammaticality,
            childes_grammar_model,
            childes_grammar_model_tokenizer,
            eval_batch_size,
        )
        s_gec = score_in_batches(
            utterances,
            compute_scores_gec,
            gec_model,
            gec_model_tokenizer,
            eval_batch_size,
        )

        all_scores_childes.extend(s_childes)
        all_scores_gec.extend(s_gec)
        all_lengths.extend(lengths)
        all_utts.extend(utterances)

    # --- Feature metrics (turn-level + set-level) ---
    # turn_df: one row per utterance, columns = per-utterance feature values
    # agg:     dict of aggregated (mean) + set-level metrics
    turn_df, feat_agg = compute_feature_metrics_for_utts(all_utts, feature_extractor, sent_model, feature_list)

    # --- Build and save the per-utterance DataFrame ---
    # Core columns: model, utterance, length, grammaticality scores
    n = len(all_utts)
    utts_df = pd.DataFrame(
        {
            "utterance": all_utts,
            "token_length": all_lengths[:n],
            "grammaticality_childes_score": all_scores_childes[:n],
            "grammaticality_gec_score": all_scores_gec[:n],
        }
    )

    # Append turn-level feature columns if available and lengths align
    if not turn_df.empty:
        if len(turn_df) == n:
            turn_df = turn_df.reset_index(drop=True)
            utts_df = pd.concat([utts_df, turn_df], axis=1)
        else:
            print(
                f"[Warning] turn_df has {len(turn_df)} rows but expected {n}; skipping per-utterance feature merge.",
                flush=True,
            )

    # Save per-utterance CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_utts_csv)), exist_ok=True)
    utts_df.to_csv(output_utts_csv, index=False)
    print(f"[Saved] per-utterance results → {output_utts_csv}", flush=True)

    # --- Aggregate summary results ---
    results = {
        "grammaticality_childes": float(np.mean(all_scores_childes)) if all_scores_childes else np.nan,
        "grammaticality_gec": float(np.mean(all_scores_gec)) if all_scores_gec else np.nan,
        "mean_length": float(np.mean(all_lengths)) if all_lengths else np.nan,
        "num_generated_sentences": int(total_generated),
        "num_scored_sentences": int(len(all_scores_childes)),
        "entropy_reg": compute_entropy_reg(all_utts),
    }
    results.update(feat_agg)

    print(
        f"\n[{model_path}] "
        f"childes={results['grammaticality_childes']:.3f} | "
        f"gec={results['grammaticality_gec']:.3f} | "
        f"mean_len={results['mean_length']:.2f} | "
        f"scored={results['num_scored_sentences']}"
    )
    return results


def eval_models(args):
    # Heavy imports deferred until all pre-flight checks have passed
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from pkg.rlhf.eval.fast_eval import (
        build_feature_list,
    )
    from pkg.rlhf.eval.gen_util import FeatureExtractor
    from pkg.rlhf.eval.grammar_util import (
        load_childes_grammar_model,
        load_gec_model,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.gen_seed)
    output_path = os.path.abspath(args.output_csv)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load evaluation models once
    childes_grammar_model, childes_grammar_model_tokenizer = load_childes_grammar_model(args.eval_model_path)
    gec_model, gec_model_tokenizer = load_gec_model()

    # Load feature resources once
    word_info = pd.read_csv(args.word_info_path)
    func_info = pd.read_csv(args.func_info_path)
    sent_model = SentenceTransformer(args.sent_model_path)
    feature_extractor = FeatureExtractor(word_info, func_info, embedding_model=sent_model)

    feature_list = build_feature_list(args.fea_set)

    all_results = []
    skipped = []

    for model_path in args.model_paths:
        model_path = os.path.abspath(model_path)
        print(f"\nEvaluating model: {model_path}")

        model_basename = os.path.basename(model_path)

        # ---- Utterance CSV naming ----
        if getattr(args, "baseline", False):
            output_utts_csv = args.output_utts_csv
        else:
            utts_csv_base, utts_csv_ext = os.path.splitext(args.output_utts_csv)
            output_utts_csv = f"{utts_csv_base}_{model_basename}{utts_csv_ext}"

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.eval()

            extra = eval_grammaticality_and_features(
                model=model,
                tokenizer=tokenizer,
                childes_grammar_model=childes_grammar_model,
                childes_grammar_model_tokenizer=childes_grammar_model_tokenizer,
                gec_model=gec_model,
                gec_model_tokenizer=gec_model_tokenizer,
                feature_extractor=feature_extractor,
                sent_model=sent_model,
                feature_list=feature_list,
                model_path=model_path,
                output_utts_csv=output_utts_csv,
                num_batches=args.num_batches,
                batch_size=args.batch_size,
                output_max_length=args.output_max_length,
                eval_batch_size=args.eval_batch_size,
            )

            results = {"model": model_path}
            results.update(extra)
            all_results.append(results)

        except Exception as e:
            print(f"Error while evaluating {model_path}: {e}")
            skipped.append(model_path)
            continue

    if not all_results:
        print("\nNo valid results produced. Nothing to save.")
        print(f"Skipped paths: {skipped}")
        return

    df = pd.DataFrame(all_results).set_index("model")

    # ---- Final results CSV naming ----
    if getattr(args, "baseline", False):
        output_csv = output_path
    else:
        output_csv_base, output_csv_ext = os.path.splitext(output_path)
        output_csv = f"{output_csv_base}_{model_basename}{output_csv_ext}"

    df.to_csv(output_csv, index=True, index_label="model")

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
        print("\nSummary:\n", df[cols_to_show])
    else:
        print("\nResults saved, but no summary columns available.")

    print(f"\nSkipped: {skipped}")
    print(f"Saved results to: {output_csv}")


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--gen_seed", type=int, default=42)
    p.add_argument("--model_paths", type=str, nargs="+", required=True)
    p.add_argument("--eval_model_path", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--num_batches", type=int, default=200)
    p.add_argument("--output_max_length", type=int, default=DEFAULT_MAX_GENERATION_LEN)
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
        help="Number of utterances per batch when scoring grammaticality.",
    )

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

    p.add_argument(
        "--output_utts_csv",
        type=str,
        default="utterances.csv",
        help=(
            "Base path for per-utterance CSVs (one file per model). "
            "The model basename is appended automatically, e.g. utterances_checkpoint-500.csv"
        ),
    )

    p.add_argument("--output_csv", type=str, default="results.csv")
    p.add_argument("--baseline", action="store_true", help="If set, do not modify output filenames or directories.")
    p.add_argument("--skip_existing", action="store_true", help="If set, skip_existing generations.")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Pre-flight: check all model paths exist before importing heavy libs
    invalid = [p for p in args.model_paths if not os.path.isdir(os.path.abspath(p))]
    if invalid:
        for p in invalid:
            print(f"Skipping non-existing checkpoint path: {p}")
        args.model_paths = [p for p in args.model_paths if os.path.isdir(os.path.abspath(p))]
        if not args.model_paths:
            print("No valid model paths found. Exiting.")
            exit(1)

    # Pre-flight: check output_utts_csv existence before importing heavy libs
    if args.skip_existing:
        if args.baseline:
            if Path(args.output_utts_csv).exists():
                print(f"WARNING: Output csv already exists: {args.output_utts_csv}")
                exit(1)
        else:
            for model_path in args.model_paths:
                model_basename = os.path.basename(os.path.abspath(model_path))
                utts_csv_base, utts_csv_ext = os.path.splitext(args.output_utts_csv)
                output_utts_csv = f"{utts_csv_base}_{model_basename}{utts_csv_ext}"
                if Path(output_utts_csv).exists():
                    print(f"WARNING: Output csv already exists: {output_utts_csv}")
                    exit(1)

    eval_models(args)
