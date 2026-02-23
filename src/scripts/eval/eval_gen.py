import argparse
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from pkg.rlhf.eval.gen_util import FeatureExtractor, SemEnt, preprocess  # adjust if needed
from pkg.rlhf.eval.grammar_util import (
    compute_scores_childes_grammaticality,
    compute_scores_gec,
    load_childes_grammar_model,
    load_gec_model,
)
from pkg.rlhf.utilities import DEFAULT_MAX_GENERATION_LEN, DEFAULT_MIN_GENERATION_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Grammaticality scoring
# ----------------------------
def compute_entropy_reg(utterances: list[str]) -> float:
    """entropy_reg = - sum_w p(w) * log(p(w))
    where p(w) = count(w) / total_words over all generated utterances.
    Uses natural log.
    """
    if not utterances:
        return float("nan")

    tokens = []
    for u in utterances:
        if not u:
            continue
        tokens.extend(u.lower().split())

    total = len(tokens)
    if total == 0:
        return float("nan")

    counts = Counter(tokens)
    probs = np.fromiter((c / total for c in counts.values()), dtype=np.float64)

    return float(-(probs * np.log(probs)).sum())


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
# Feature-based metrics
# ----------------------------
def compute_feature_metrics_for_utts(
    utterances: list[str],
    feature_extractor: FeatureExtractor,
    sent_model: SentenceTransformer,
    feature_list: list[str],
):
    """Compute turn-level metrics (word/syn) per utterance and aggregate.
    Also compute set-level diversity metrics (lemma_div/dep_div/sem_div) and semantic entropy.

    Returns
    -------
    turn_df : pd.DataFrame
        One row per utterance, one column per turn-level feature.
        Empty DataFrame if no utterances or no per-utterance features apply.
    agg : dict
        Aggregated (mean) values of turn-level features, plus set-level diversity metrics.

    """
    print(f"[Features] computing features for {len(utterances)} utterances", flush=True)
    if len(utterances) == 0:
        return pd.DataFrame(), {}

    cleaned = [preprocess(u) for u in utterances]

    df = pd.DataFrame(
        {
            "cleaned": cleaned,
            "speaker": ["CHI"] * len(cleaned),
        }
    )

    print("[Features] spaCy parsing + dependency graphs", flush=True)
    doc_lst, lemma_lst, dep_graphs = feature_extractor.extract_doc(df["cleaned"], feature_list)
    print("[Features] sentence embeddings", flush=True)
    word_vectors = feature_extractor.extract_vec(df["cleaned"], feature_list)

    # --- Turn-level features (per utterance) ---
    per_utt_features = [
        "freq",
        "conc",
        "word_len",
        "sent_len",
        "ttr",
        "distinct_2",
        "distinct_3",
        "tree_depth",
        "clause",
        "lex_den",
        "func_den",
        "func_den_new",
        "pp_den",
        "non_word_rate",
        "non_word_type_rate",
    ]
    per_utt_features = [f for f in per_utt_features if f in feature_list]

    print(f"[Features] computing turn-level features for {len(cleaned)} utterances", flush=True)
    rows = []
    for i in range(len(cleaned)):
        if i % 100 == 0:
            print(f"[Features] turn-level features {i}/{len(cleaned)}", flush=True)

        rows.append(
            feature_extractor.get_turn_fea(
                df["cleaned"].iloc[i],
                doc_lst[i] if i < len(doc_lst) else "",
                lemma_lst[i] if i < len(lemma_lst) else "",
                per_utt_features,
                ["VERB", "NOUN", "ADV", "ADJ", "PROPN"],  # content_POS
                ["VERB", "NOUN", "PROPN", "ADV", "ADJ", "PRON", "INTJ"],  # target_POS
                "word",  # func_header
            )
        )

    # turn_df: one row per utterance, one column per turn-level feature
    turn_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Aggregate means for the summary results dict
    agg = {}
    for col in turn_df.columns:
        if pd.api.types.is_numeric_dtype(turn_df[col]):
            agg[f"{col}_mean"] = float(turn_df[col].mean())

    # --- Set-level diversity features ---
    indices = list(range(len(cleaned)))

    div_candidates = [f for f in ["lemma_div", "dep_div", "sem_div"] if f in feature_list]
    if div_candidates:
        div_vals = feature_extractor.get_div_fea(feature_list, indices, lemma_lst, dep_graphs, word_vectors)
        if isinstance(div_vals, dict):
            for k, v in div_vals.items():
                try:
                    agg[f"{k}_set"] = float(v)
                except Exception:
                    pass
        elif isinstance(div_vals, (list, tuple, np.ndarray)):
            for j, v in enumerate(div_vals):
                agg[f"div_{j}_set"] = float(v)

    # --- Semantic entropy (set-level) ---
    try:
        semEnt = SemEnt(word_vectors, indices=indices)
        sem_ent, k = semEnt.compute_diversity()
        agg["sem_ent_set"] = float(sem_ent)
        agg["sem_ent_k_set"] = float(k)
    except Exception:
        pass

    agg["num_scored_utts"] = int(len(cleaned))
    agg["mean_chars"] = float(np.mean([len(u) for u in utterances]))

    # turn_df may have fewer columns than per_utt_features if some weren't computed;
    # we return it as-is and let the caller align it with utterances/scores.
    return turn_df, agg


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
            "model": [model_path] * n,
            "utterance": all_utts,
            "length": all_lengths[:n],
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


# ----------------------------
# Orchestration
# ----------------------------
def build_feature_list(fea_set):
    fea_dict = {
        "word": [
            "freq",
            "conc",
            "word_len",
            "sent_len",
            "ttr",
            "non_word_type_rate",
            "non_word_rate",
            "distinct_2",
            "distinct_3",
        ],
        "syn": ["tree_depth", "clause", "pp_den", "lex_den", "func_den", "func_den_new"],
        "div": ["lemma_div", "dep_div", "sem_div"],
        "semEnt": ["sem_ent"],
    }
    flat = []
    for key in fea_set:
        flat.extend(fea_dict.get(key, []))
    seen = set()
    out = []
    for f in flat:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def eval_models(args):
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
        print(f"\nChecking model path: {model_path}")

        if not os.path.isdir(model_path):
            print("Skipping non-existing checkpoint path:", model_path)
            skipped.append(model_path)
            continue

        # Derive a per-model utterance CSV path from --output_utts_csv
        # e.g. utterances.csv → utterances_<model_basename>.csv
        model_basename = os.path.basename(model_path)
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
                eval_batch_size=args.eval_batch_size,  # now wired up
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
    df.to_csv(output_path, index=True, index_label="model")

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
    print(f"Saved results to: {output_path}")


def get_args():
    p = argparse.ArgumentParser()

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
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_models(args)
