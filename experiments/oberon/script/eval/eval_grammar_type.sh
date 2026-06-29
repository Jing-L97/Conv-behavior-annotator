#!/bin/bash
# ── positional arguments ──────────────────────────────────────────────────────
JUDGE_MODEL=$1 # LM pretraining seed

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"

TARGET_MDOEL=$MODEL_ROOT/lm/lightning_logs/$DATA_SIZE/$SEED/ckpt_huggingface_best
OUTPUT_DIR=$ROOT/results/childes/grammar_val/$JUDGE_MODEL

mkdir -p $OUTPUT_DIR

python -u $SCRIPT_ROOT/eval/benchmark_llm_judge_fewshot.py \
    --data_dir $DATA_ROOT/grammar_error_childes \
    --output_dir $OUTPUT_DIR \
    --judge_model $JUDGE_MODEL \
    --dtype bfloat16 \
    --device auto \
    --batch_size 32 \
    --n_grammatical_fewshot 5 \
    --limit 10000

