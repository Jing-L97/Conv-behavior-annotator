#!/bin/bash
# ── positional arguments ──────────────────────────────────────────────────────
SEED=$1 # LM pretraining seed
DATA_SIZE=$2
# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
DATA_ROOT=$ROOT/"datasets"
MODEL_ROOT=$ROOT/"models"
TARGET_MDOEL=$MODEL_ROOT/lm/lightning_logs/$DATA_SIZE/$SEED/ckpt_huggingface_best
OUTPUT_DIR=$ROOT/results/baseline/$DATA_SIZE/$SEED

mkdir -p $OUTPUT_DIR

python -u $SCRIPT_ROOT/eval/eval_grammar.py \
    --model_paths $TARGET_MDOEL \
    --eval_data_dir $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --skip_existing 
    