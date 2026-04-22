#!/bin/bash
# ── positional arguments ──────────────────────────────────────────────────────
SEED=$1 # LM pretraining seed
DATA_SIZE=$2
GEN_SEED=$3
# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"

TARGET_MDOEL=$MODEL_ROOT/lm/lightning_logs/$DATA_SIZE/$SEED/ckpt_huggingface_best
OUTPUT_DIR=$ROOT/results/baseline/$DATA_SIZE/$SEED/$GEN_SEED

mkdir -p $OUTPUT_DIR

python -u $SCRIPT_ROOT/eval/eval_gen.py \
    --model_paths $TARGET_MDOEL \
    --eval_model_path $MODEL_ROOT/grammar_eval/version_19 \
    --word_info_path $DATA_ROOT/evaluation_data/gen/word_info.csv \
    --func_info_path $DATA_ROOT/evaluation_data/gen/func_info.csv \
    --output_utts_csv $OUTPUT_DIR/utt.csv \
    --output_csv $OUTPUT_DIR/result.csv \
    --skip_existing \
    --gen_seed $GEN_SEED \
    --batch_size 50 \
    --num_batches 200 \
    --baseline


    