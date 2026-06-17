#!/usr/bin/env python3
"""Benchmark the LLM-as-judge against human labels with balanced few-shot prompting.

Protocol (Option C -- one structured call per utterance):
    - Load the human-annotated child utterances (only [CHI] rows that have labels;
      every such row is ungrammatical).
    - Sample a BALANCED few-shot pool: up to K demos per category (rare categories
      first), each demo rendered as a full per-category true/false checklist. The
      remaining utterances are the TEST set (no sentence is both demo and test).
    - For each test utterance, send ONE fixed-prefix prompt (definitions + the same
      few-shot block + the utterance last) and ask the judge to mark true/false for
      every category. The few-shot block is built once and reused on every call
      (reproducible + prefix-cacheable by Ollama).
    - Score per-category precision / recall / F1 against the gold labels and write
      two artifacts: a per-row predictions CSV and a per-category stats CSV.

Reuses the existing judge transport + taxonomy (pkg.rlhf.eval.llm_judge) and the
data/label/metric utilities (pkg.rlhf.eval.error_type_classifier). Only the
sampling + sweep + IO glue is new.

Example
-------
    python -m scripts.eval.benchmark_llm_judge_fewshot \
        --data_dir   ../sample_data \
        --output_dir runs/judge_fewshot \
        --judge_model qwen3:8b --k_per_category 5

Dry run first (e.g. 20 test utterances):
    python -m scripts.eval.benchmark_llm_judge_fewshot --data_dir ../sample_data --limit 20
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from pkg.rlhf.eval.error_type_classifier import (
    ERROR_TYPE_LABELS,
    LABEL_ALIASES,
    compute_multilabel_metrics,
    encode_targets,
    load_annotated_dir,
    multihot_to_labels,
)
from pkg.rlhf.eval.llm_judge import (
    TAXONOMY,
    JudgeConfig,
    build_checklist_messages,
    call_judge_once,
    parse_checklist_json,
    render_fewshot_block,
)

# Judge taxonomy is keyed by "progressive"; the data uses "present_progressive".
JUDGE_KEY_FOR_DATA_LABEL = {"present_progressive": "progressive"}


def build_definitions() -> dict:
    """Bridge the judge TAXONOMY definitions onto the data's label spelling."""
    defs = {}
    for label in ERROR_TYPE_LABELS:
        judge_key = JUDGE_KEY_FOR_DATA_LABEL.get(label, label)
        spec = TAXONOMY.get(judge_key)
        defs[label] = spec["definition"] if spec else ""
    return defs


# ----------------------------
# Balanced few-shot sampling
# ----------------------------
def sample_balanced_fewshot(label_lists, k=5, seed=1, n_multi_error=2):
    """Pick ~k demo sentences per category (rare-first), plus a few multi-error demos.

    Returns (fewshot_idx sorted list, test_idx sorted list). Each sentence appears
    in at most one set. ``label_lists`` is the per-row list of gold canonical labels.
    """
    rng = np.random.default_rng(seed)
    n = len(label_lists)
    counts = {lab: 0 for lab in ERROR_TYPE_LABELS}
    for lst in label_lists:
        for lab in lst:
            if lab in counts:
                counts[lab] += 1

    chosen: set[int] = set()
    # rarest categories first so their (few) sentences are not consumed by commons.
    for cat in sorted(ERROR_TYPE_LABELS, key=lambda c: counts[c]):
        already = sum(1 for i in chosen if cat in label_lists[i])
        need = k - already
        if need <= 0:
            continue
        cands = [i for i in range(n) if i not in chosen and cat in label_lists[i]]
        rng.shuffle(cands)
        for i in cands[:need]:
            chosen.add(i)

    # ensure a couple of explicit multi-error demos are present.
    have_multi = sum(1 for i in chosen if len(set(label_lists[i])) >= 2)
    if have_multi < n_multi_error:
        cands = [i for i in range(n) if i not in chosen and len(set(label_lists[i])) >= 2]
        rng.shuffle(cands)
        for i in cands[: n_multi_error - have_multi]:
            chosen.add(i)

    fewshot_idx = sorted(chosen)
    test_idx = [i for i in range(n) if i not in chosen]
    return fewshot_idx, test_idx


# ----------------------------
# One checklist judgment (with retries; reuses call_judge_once)
# ----------------------------
def judge_one(utterance, definitions, fewshot_block, config, client):
    messages = build_checklist_messages(utterance, ERROR_TYPE_LABELS, definitions, fewshot_block)
    last_raw = ""
    for attempt in range(1, config.max_retries + 1):
        try:
            last_raw = call_judge_once(messages, config, client=client)
            preds, ok = parse_checklist_json(last_raw, ERROR_TYPE_LABELS, aliases=LABEL_ALIASES)
            if ok:
                return preds, last_raw, True
        except Exception as exc:  # noqa: BLE001 - judge must be robust
            last_raw = f"{type(exc).__name__}: {exc}"
        if attempt < config.max_retries:
            time.sleep(config.retry_backoff * attempt)
    return {lab: 0 for lab in ERROR_TYPE_LABELS}, last_raw, False


# ----------------------------
# Main
# ----------------------------
def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_annotated_dir(args.data_dir, utt_column=args.utt_column, speaker_code=args.speaker_code)
    gold_matrix, label_lists = encode_targets(df)
    utterances = df[args.utt_column].astype(str).str.strip().tolist()

    fewshot_idx, test_idx = sample_balanced_fewshot(
        label_lists, k=args.k_per_category, seed=args.seed, n_multi_error=args.n_multi_error
    )
    print(f"[split] few-shot demos: {len(fewshot_idx)} | test utterances: {len(test_idx)}")

    # Build the fixed few-shot block ONCE (reused on every call).
    fewshot_examples = [(utterances[i], label_lists[i]) for i in fewshot_idx]
    fewshot_block = render_fewshot_block(fewshot_examples, ERROR_TYPE_LABELS)
    definitions = build_definitions()

    # Persist the demos used (transparency / reproducibility).
    pd.DataFrame(
        {"utterance": [utterances[i] for i in fewshot_idx], "labels": [", ".join(label_lists[i]) for i in fewshot_idx]}
    ).to_csv(os.path.join(args.output_dir, "fewshot_used.csv"), index=False)

    if args.limit:
        test_idx = test_idx[: args.limit]
        print(f"[limit] truncated test set to {len(test_idx)} utterances.")

    config = JudgeConfig(model=args.judge_model, host=args.ollama_host, temperature=args.temperature)
    client = None if args.dry_run else __import__("pkg.rlhf.eval.llm_judge", fromlist=["_get_client"])._get_client(config)

    # --- Inference sweep ---
    rows = []
    pred_matrix = np.zeros((len(test_idx), len(ERROR_TYPE_LABELS)), dtype=int)
    for n, i in enumerate(test_idx):
        if args.dry_run:
            preds, raw, ok = {lab: 0 for lab in ERROR_TYPE_LABELS}, "(dry-run: no call)", True
        else:
            preds, raw, ok = judge_one(utterances[i], definitions, fewshot_block, config, client)

        pred_vec = np.array([preds[lab] for lab in ERROR_TYPE_LABELS], dtype=int)
        pred_matrix[n] = pred_vec
        gold_set = set(label_lists[i])
        pred_set = set(multihot_to_labels(pred_vec))

        row = {"row_id": int(df.index[i]), "utterance": utterances[i]}
        row["gold_labels"] = ", ".join(label_lists[i])
        row["judge_labels"] = ", ".join(sorted(pred_set, key=ERROR_TYPE_LABELS.index))
        for j, lab in enumerate(ERROR_TYPE_LABELS):
            row[f"gold_{lab}"] = int(gold_matrix[i, j])
            row[f"judge_{lab}"] = int(pred_vec[j])
        row["exact_match"] = int(gold_set == pred_set)
        row["judge_ok"] = int(ok)
        row["judge_raw"] = raw
        rows.append(row)

        if (n + 1) % args.log_every == 0 or (n + 1) == len(test_idx):
            n_bad = sum(not r["judge_ok"] for r in rows)
            print(f"[judge] {n + 1}/{len(test_idx)} | malformed so far: {n_bad}", flush=True)

    pred_df = pd.DataFrame(rows)
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # --- Per-category scoring ---
    gold_test = gold_matrix[test_idx]
    metrics = compute_multilabel_metrics(gold_test, pred_matrix)

    stat_rows = []
    for lab in ERROR_TYPE_LABELS:
        pc = metrics["per_class"][lab]
        stat_rows.append({"label": lab, **pc})
    for agg in ("micro_f1", "macro_f1", "samples_f1", "macro_f1_nonsparse"):
        stat_rows.append({"label": agg, "precision": "", "recall": "", "f1": metrics[agg], "support": ""})
    stats_df = pd.DataFrame(stat_rows)
    stats_path = os.path.join(args.output_dir, "per_category_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    print("\n=== Per-category stats ===")
    print(stats_df.to_string(index=False))
    print(
        f"\nmicro_f1={metrics['micro_f1']:.3f} macro_f1={metrics['macro_f1']:.3f} "
        f"samples_f1={metrics['samples_f1']:.3f} (exact_match={pred_df['exact_match'].mean():.3f})"
    )
    print(f"[saved] {pred_path}\n[saved] {stats_path}\n[saved] fewshot_used.csv -> {args.output_dir}")


def get_args():
    p = argparse.ArgumentParser(description="Few-shot LLM-as-judge benchmark vs human labels (checklist mode).")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="runs/judge_fewshot")
    p.add_argument("--utt_column", type=str, default="transcript_clean")
    p.add_argument("--speaker_code", type=str, default="[CHI]")
    p.add_argument("--k_per_category", type=int, default=5, help="Balanced few-shot demos per category.")
    p.add_argument("--n_multi_error", type=int, default=2, help="Extra demos guaranteed to be multi-label.")
    p.add_argument("--judge_model", type=str, default="qwen3:8b")
    p.add_argument("--ollama_host", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Cap #test utterances (0 = all).")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--dry_run", action="store_true", help="Build prompts/split but make no Ollama calls.")
    args = p.parse_args()
    if args.speaker_code == "":
        args.speaker_code = None
    return args


if __name__ == "__main__":
    run(get_args())
