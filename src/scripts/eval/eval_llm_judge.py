#!/usr/bin/env python3
"""Stage-2 eval method: LLM-as-judge grammatical-error annotation (Ollama).

Reads a CSV of generated utterances, runs an open-weights LLM judge (served via
Ollama) over each row, and appends the judge's per-category error columns *in
place* to the input rows. Mirrors the stage-2 pattern of ``eval_childes.py``:

    input CSV  --(judge per row)-->  input CSV + judge_* columns   (--output_csv)
                                     aggregate summary (one row)    (--summary_csv)

All judge logic, the taxonomy spec, and the pipeline-format adapter live in
``pkg.rlhf.eval.llm_judge`` -- this file only handles I/O, batching/logging and
the CLI. Rows are never dropped (empties are routed to ``unintelligible``) so the
output stays row-aligned with the input for later human-vs-judge kappa.

Swap the judge model:   --judge_model qwen2.5:7b   (or llama3.1:8b, etc.)
Edit the taxonomy:      edit pkg/rlhf/eval/llm_judge.py (TAXONOMY) -- nothing here.

Example
-------
    python -m scripts.eval.eval_llm_judge \
        --input_csv  scripts/eval/sample_judge_input.csv \
        --utt_column utterance \
        --context_column prev_transcript_clean \
        --judge_model qwen3:8b \
        --output_csv  sample_judge_output.csv \
        --summary_csv sample_judge_summary.csv
"""

import argparse
import os

import pandas as pd

from pkg.rlhf.eval.llm_judge import (
    JUDGE_COLUMNS,
    JudgeConfig,
    aggregate_summary,
    judge_utterance,
    result_to_row,
)


# ----------------------------
# Per-row judging
# ----------------------------
def judge_dataframe(
    df: pd.DataFrame,
    utt_column: str,
    config: JudgeConfig,
    context_column: str | None = None,
    log_every: int = 25,
) -> tuple[pd.DataFrame, list]:
    """Run the judge over every row of ``df``; return (annotated df, results).

    Rows are kept 1:1 with the input (empties routed to ``unintelligible``). A
    stable ``row_id`` (the original index) is added as the join key for kappa.
    """
    if utt_column not in df.columns:
        raise ValueError(f"Column '{utt_column}' not found. Available columns: {list(df.columns)}")
    if context_column is not None and context_column not in df.columns:
        print(f"[Warning] context column '{context_column}' not found; proceeding without context.")
        context_column = None

    df = df.reset_index(drop=True).copy()
    if "row_id" not in df.columns:
        df.insert(0, "row_id", df.index)

    # Reuse one Ollama client across the whole run.
    from pkg.rlhf.eval.llm_judge import _get_client

    client = _get_client(config)

    results = []
    rows = []
    n = len(df)
    for i, record in df.iterrows():
        utterance = "" if pd.isna(record[utt_column]) else str(record[utt_column])
        context = None
        if context_column is not None and not pd.isna(record[context_column]):
            context = str(record[context_column])

        result = judge_utterance(utterance, config, context=context, client=client)
        results.append(result)
        rows.append(result_to_row(result))

        if (i + 1) % log_every == 0 or (i + 1) == n:
            n_bad = sum(not r.ok for r in results)
            print(f"[judge] {i + 1}/{n} rows | malformed so far: {n_bad}", flush=True)

    judge_df = pd.DataFrame(rows, columns=JUDGE_COLUMNS)
    annotated = pd.concat([df.reset_index(drop=True), judge_df.reset_index(drop=True)], axis=1)
    return annotated, results


# ----------------------------
# Entry point
# ----------------------------
def eval_csv(args) -> None:
    df = pd.read_csv(args.input_csv)
    print(f"[Read] {len(df)} rows from '{args.input_csv}' (utt column='{args.utt_column}')")

    if args.debug:
        df = df.head(args.debug_rows)
        print(f"[Debug] Truncated to {len(df)} rows.")

    config = JudgeConfig(
        model=args.judge_model,
        host=args.ollama_host,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    print(f"[Judge] model='{config.model}' host='{config.host or 'default'}' temp={config.temperature}")

    annotated, results = judge_dataframe(
        df,
        utt_column=args.utt_column,
        config=config,
        context_column=args.context_column,
    )

    # --- Save annotated per-row CSV (input + judge_* columns) ---
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    annotated.to_csv(args.output_csv, index=False)
    print(f"[Saved] annotated rows -> {args.output_csv}", flush=True)

    # --- Save aggregate summary (one row, additive scalar keys) ---
    summary = aggregate_summary(results)
    summary_path = args.summary_csv or (os.path.splitext(args.output_csv)[0] + "_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[Saved] summary -> {summary_path}", flush=True)

    # --- Console summary ---
    n = summary.get("judge_num_scored", 0)
    print(
        f"\nScored {n} rows | malformed={summary.get('judge_num_malformed', 0)} | "
        f"grammatical={summary.get('judge_rate_grammatical', float('nan')):.3f} | "
        f"unintelligible={summary.get('judge_rate_unintelligible', float('nan')):.3f}"
    )


# ----------------------------
# CLI
# ----------------------------
def get_args():
    p = argparse.ArgumentParser(description="LLM-as-judge grammatical-error annotation over a CSV of utterances.")

    # --- Input ---
    p.add_argument("--input_csv", type=str, required=True, help="CSV of utterances to annotate.")
    p.add_argument("--utt_column", type=str, default="utterance", help="Column holding the utterance text.")
    p.add_argument(
        "--context_column",
        type=str,
        default=None,
        help="Optional column with the preceding caregiver turn (context only, e.g. 'prev_transcript_clean').",
    )

    # --- Judge model (swappable) ---
    p.add_argument("--judge_model", type=str, default="qwen3:8b", help="Ollama model tag for the judge.")
    p.add_argument("--ollama_host", type=str, default=None, help="Ollama host URL (default: localhost:11434).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_retries", type=int, default=3)

    # --- Outputs ---
    p.add_argument("--output_csv", type=str, default="judge_annotated.csv", help="Input rows + judge_* columns.")
    p.add_argument("--summary_csv", type=str, default=None, help="Aggregate summary CSV (default: <output>_summary.csv).")

    # --- Debug ---
    p.add_argument("--debug", action="store_true", help="If set, only annotate the first --debug_rows rows.")
    p.add_argument("--debug_rows", type=int, default=10)

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_csv(args)
