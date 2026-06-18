#!/usr/bin/env python3
"""Benchmark the LLM-as-judge against human labels with balanced few-shot prompting.

Protocol (Option C -- one structured call per utterance):
    - Load the human-annotated child utterances. By default only [CHI] rows that
      have labels are kept; with --include_unlabeled, empty-label rows are kept
      and treated as grammatical/no-error examples.
    - Sample a BALANCED few-shot pool: up to K demos per category (rare categories
      first), each demo rendered as a full per-category true/false checklist. The
      remaining utterances are the TEST set (no sentence is both demo and test).
    - For each test utterance, send ONE fixed-prefix prompt (definitions + the same
      few-shot block + the utterance last) and ask the judge to mark true/false for
      every category. HuggingFace inference can batch these prompts for GPU
      throughput on a cluster.
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
        --judge_model Qwen/Qwen3-8B --k_per_category 5 --batch_size 16

Dry run first (e.g. 20 test utterances):
    python -m scripts.eval.benchmark_llm_judge_fewshot --data_dir ../sample_data --limit 20
"""

import argparse
import dataclasses
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
    HFJudgeConfig,
    build_checklist_messages,
    call_judge_hf_batch,
    load_hf_judge,
    parse_checklist_json_with_rationale,
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
def sample_balanced_fewshot(label_lists, k=5, seed=1, n_multi_error=2, n_grammatical=5):
    """Pick ~k demos per category, plus multi-error and no-error demos.

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

    if n_grammatical > 0:
        cands = [i for i in range(n) if i not in chosen and not label_lists[i]]
        rng.shuffle(cands)
        for i in cands[:n_grammatical]:
            chosen.add(i)

    fewshot_idx = sorted(chosen)
    test_idx = [i for i in range(n) if i not in chosen]
    return fewshot_idx, test_idx


# ----------------------------
# One checklist judgment (with retries; reuses HF transport)
# ----------------------------
def _retry_config(config: HFJudgeConfig) -> HFJudgeConfig:
    if not config.retry_do_sample:
        return config
    return dataclasses.replace(config, do_sample=True, temperature=config.retry_temperature)


def _format_label_summary(label_list: list[str], is_ungrammatical: bool) -> str:
    """Return the compact label summary used in the predictions CSV."""
    if label_list:
        return ", ".join(label_list)
    return "unclassified_error" if is_ungrammatical else "grammatical"


def _judge_batch(
    utterance_batch: list[str],
    definitions: dict[str, str],
    fewshot_block: str,
    config: HFJudgeConfig,
    model: object,
    tokenizer: object,
) -> list[tuple[dict[str, int], str, bool, str, bool]]:
    """Judge a batch and retry malformed outputs with sampled decoding."""
    messages_batch = [
        build_checklist_messages(utterance, ERROR_TYPE_LABELS, definitions, fewshot_block)
        for utterance in utterance_batch
    ]
    results = []
    try:
        raw_batch = call_judge_hf_batch(messages_batch, model, tokenizer, config)
    except Exception as exc:  # noqa: BLE001 - judge must be robust
        raw_batch = [f"{type(exc).__name__}: {exc}"] * len(messages_batch)

    for raw in raw_batch:
        preds, rationale, is_ungrammatical, ok = parse_checklist_json_with_rationale(
            raw, ERROR_TYPE_LABELS, aliases=LABEL_ALIASES
        )
        results.append([preds, rationale, is_ungrammatical, raw, ok])

    for attempt in range(2, config.max_retries + 1):
        bad_positions = [pos for pos, (_, _, _, _, ok) in enumerate(results) if not ok]
        if not bad_positions:
            break
        time.sleep(config.retry_backoff * (attempt - 1))
        retry_messages = [messages_batch[pos] for pos in bad_positions]
        retry_config = _retry_config(config)
        try:
            retry_raws = call_judge_hf_batch(retry_messages, model, tokenizer, retry_config)
        except Exception as exc:  # noqa: BLE001 - judge must be robust
            retry_raws = [f"{type(exc).__name__}: {exc}"] * len(retry_messages)
        for pos, raw in zip(bad_positions, retry_raws, strict=True):
            preds, rationale, is_ungrammatical, ok = parse_checklist_json_with_rationale(
                raw, ERROR_TYPE_LABELS, aliases=LABEL_ALIASES
            )
            results[pos] = [preds, rationale, is_ungrammatical, raw, ok]

    return [
        (preds if ok else {lab: 0 for lab in ERROR_TYPE_LABELS}, rationale, is_ungrammatical if ok else False, raw, ok)
        for preds, rationale, is_ungrammatical, raw, ok in results
    ]


# ----------------------------
# Main
# ----------------------------
def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_annotated_dir(
        args.data_dir,
        utt_column=args.utt_column,
        speaker_code=args.speaker_code,
        require_labels=not args.include_unlabeled,
    )
    gold_matrix, label_lists = encode_targets(df)
    utterances = df[args.utt_column].astype(str).str.strip().tolist()

    fewshot_idx, test_idx = sample_balanced_fewshot(
        label_lists,
        k=args.k_per_category,
        seed=args.seed,
        n_multi_error=args.n_multi_error,
        n_grammatical=args.n_grammatical_fewshot,
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

    config = HFJudgeConfig(
        model=args.judge_model,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
        batch_size=max(1, args.batch_size),
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        retry_do_sample=not args.no_retry_sample,
        retry_temperature=args.retry_temperature,
        trust_remote_code=args.trust_remote_code,
    )
    if args.dry_run:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_hf_judge(config)

    # --- Inference sweep ---
    rows = []
    pred_matrix = np.zeros((len(test_idx), len(ERROR_TYPE_LABELS)), dtype=int)
    processed = 0
    next_log = args.log_every if args.log_every > 0 else len(test_idx)
    for start in range(0, len(test_idx), config.batch_size):
        batch_idx = test_idx[start : start + config.batch_size]
        if args.dry_run:
            batch_results = [
                ({lab: 0 for lab in ERROR_TYPE_LABELS}, "", False, "(dry-run: no call)", True)
                for _ in batch_idx
            ]
        else:
            batch_results = _judge_batch(
                [utterances[i] for i in batch_idx],
                definitions,
                fewshot_block,
                config,
                model,
                tokenizer,
            )

        for i, (preds, rationale, is_ungrammatical, raw, ok) in zip(batch_idx, batch_results, strict=True):
            n = processed
            pred_vec = np.array([preds[lab] for lab in ERROR_TYPE_LABELS], dtype=int)
            pred_matrix[n] = pred_vec
            gold_set = set(label_lists[i])
            pred_set = set(multihot_to_labels(pred_vec))
            gold_is_ungrammatical = bool(gold_set)

            row = {"row_id": int(df.index[i]), "utterance": utterances[i]}
            row["gold_is_ungrammatical"] = int(gold_is_ungrammatical)
            row["judge_is_ungrammatical"] = int(is_ungrammatical)
            row["gold_labels"] = _format_label_summary(label_lists[i], gold_is_ungrammatical)
            for j, lab in enumerate(ERROR_TYPE_LABELS):
                row[f"gold_{lab}"] = int(gold_matrix[i, j])
                row[f"judge_{lab}"] = int(pred_vec[j])
            row["exact_match"] = int(gold_is_ungrammatical == is_ungrammatical and gold_set == pred_set)
            row["judge_ok"] = int(ok)
            row["judge_rationale"] = rationale
            row["judge_raw"] = raw
            rows.append(row)
            processed += 1

        if processed >= next_log or processed == len(test_idx):
            n_bad = sum(not r["judge_ok"] for r in rows)
            print(f"[judge] {processed}/{len(test_idx)} | malformed so far: {n_bad}", flush=True)
            while args.log_every > 0 and next_log <= processed:
                next_log += args.log_every

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
    p.add_argument("--n_grammatical_fewshot", type=int, default=5, help="No-error demos when unlabeled rows are kept.")
    p.add_argument("--include_unlabeled", action="store_true", help="Keep empty-label rows as grammatical/no-error rows.")
    p.add_argument("--judge_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", type=str, default="bfloat16", help="HF torch dtype: auto, bfloat16, float16, or float32.")
    p.add_argument("--device", type=str, default="auto", help="HF device target, e.g. auto, cuda, cuda:0, cpu.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--do_sample", action="store_true", help="Use sampled generation for the first pass.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3 thinking mode in the chat template.")
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--retry_backoff", type=float, default=2.0)
    p.add_argument("--retry_temperature", type=float, default=0.3)
    p.add_argument("--no_retry_sample", action="store_true", help="Do not sample on malformed-output retries.")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="Cap #test utterances (0 = all).")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--dry_run", action="store_true", help="Build prompts/split but make no HF model calls.")
    args = p.parse_args()
    if args.speaker_code == "":
        args.speaker_code = None
    return args


if __name__ == "__main__":
    run(get_args())
