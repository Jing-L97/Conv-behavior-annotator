#!/bin/bash
# ── positional arguments ──────────────────────────────────────────────────────
COL=$1

# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
OUT_ROOT=$ROOT/"results/childes"


python -u $SCRIPT_ROOT/eval/eval_childes.py \
    --input_csv $DATA_ROOT/raw/conversations_min_age_10.csv \
    --utt_column $COL \
    --eval_model_path $MODEL_ROOT/grammar_eval/version_19 \
    --word_info_path $DATA_ROOT/evaluation_data/gen/word_info.csv \
    --func_info_path $DATA_ROOT/evaluation_data/gen/func_info.csv \
    --output_utts_csv $OUT_ROOT/$COL/utt.csv \
    --output_csv $OUT_ROOT/$COL/result.csv