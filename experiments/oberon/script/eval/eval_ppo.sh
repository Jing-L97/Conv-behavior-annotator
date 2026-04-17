#!/bin/bash
# ── positional arguments ──────────────────────────────────────────────────────
REWARD=$1
SEED=$2
EXP=$3
EXP_SETTING=$4
GEN_SEED=$5
# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
PPO_MODEL=$MODEL_ROOT/ppo/$EXP_SETTING/$SEED/${REWARD}_$EXP/best_reward
OUTPUT_DIR=$ROOT/results/ppo/$EXP_SETTING/$SEED/${REWARD}_$EXP/$GEN_SEED

mkdir -p $OUTPUT_DIR
python -u $SCRIPT_ROOT/eval/eval_gen.py \
    --model_paths $PPO_MODEL \
    --eval_model_path $MODEL_ROOT/grammar_eval/version_19 \
    --word_info_path $DATA_ROOT/evaluation_data/gen/word_info.csv \
    --func_info_path $DATA_ROOT/evaluation_data/gen/func_info.csv \
    --output_utts_csv $OUTPUT_DIR/utt.csv \
    --output_csv $OUTPUT_DIR/result.csv \
    --gen_seed $GEN_SEED \
    --skip_existing \
    --batch_size 50 \
    --num_batches 200

