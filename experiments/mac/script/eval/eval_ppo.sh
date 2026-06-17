#!/bin/bash
# ── positional arguments ──────────────────────────────────────────────────────
REWARD=$1
FINETUNE_SEED=$2
EXP=$3
EXP_SETTING=$4
GEN_SEED=$5
PRETRAIN_SEED=$6

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
PPO_MODEL=$MODEL_ROOT/ppo/$EXP_SETTING/$PRETRAIN_SEED/$FINETUNE_SEED/${REWARD}_$EXP/best_reward
OUTPUT_DIR=$ROOT/results/ppo/$EXP_SETTING/$PRETRAIN_SEED/$FINETUNE_SEED/${REWARD}_$EXP/$GEN_SEED

mkdir -p $OUTPUT_DIR


python -m src.scripts.eval.eval_llm_judge \
  --input_csv src/scripts/eval/sample_judge_input.csv \
  --utt_column utterance --context_column prev_transcript_clean \
  --judge_model qwen3:8b \
  --output_csv sample_judge_output.csv