#!/bin/bash

# ── positional arguments ──────────────────────────────────────────────────────
REWARD=$1
REWARD_SEED=$2
SEED=$3
EXP=$4
EXP_SETTING=$5
LM=$6

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
REWARD_PATH=$MODEL_ROOT/reward/$REWARD_SEED/$REWARD 
if [[ "$REWARD" == "topline" ]]; then
    REWARD_PATH=$MODEL_ROOT/reward/$REWARD
fi

mkdir -p $MODEL_ROOT/ppo/$REWARD_$EXP/$SEED

python -u $SCRIPT_ROOT/train/train_ppo.py \
    --policy_model $MODEL_ROOT/lm/lightning_logs/$LM/ckpt_huggingface_best/ \
    --value_model $REWARD_PATH \
    --steps 6000 \
    --target 6 \
    --lm_data_path $DATA_ROOT/raw/caregiver_utterances_train_1000000.0_words.txt \
    --lm_val_data_path $DATA_ROOT/raw/caregiver_utterances_val_1000000.0_words.txt \
    --grammar_eval_model_path $MODEL_ROOT/grammar_eval/version_19 \
    --entropy_reg_coef 0.001 \
    --length_reward_coef 0 \
    --lm_loss_coef 0.001 \
    --exp_name $REWARD"_"$EXP \
    --eval_data_dir $DATA_ROOT \
    --output_dir $MODEL_ROOT/ppo/$EXP_SETTING/$SEED \
    --wandb_dir $ROOT \
    --skip_existing \
    --seed $SEED