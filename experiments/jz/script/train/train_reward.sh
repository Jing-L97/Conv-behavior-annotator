#!/bin/bash
# ── positional arguments ──────────────────────────────────────────────────────
COL_NAME=$1
SEED=$2
# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
# ── conditional arguments ─────────────────────────────────────────────────────
EXTRA_ARGS=""
if [[ "$COL_NAME" == continuous_* ]]; then
    COL_NAME="${COL_NAME#continuous_}"
elif [[ "$COL_NAME" == align_* ]]; then
    EXTRA_ARGS="--apply_binary"
fi
python $SCRIPT_ROOT/train/train_reward.py \
    --data_path $DATA_ROOT/annotated/conversations_min_age_10.csv \
    --model_name_or_path microsoft/deberta-v3-xsmall \
    --output_dir $MODEL_ROOT/reward/$SEED/$COL_NAME \
    --reward_column_name $COL_NAME \
    --wandb_dir $ROOT \
    --skip_existing \
    --seed $SEED \
    $EXTRA_ARGS