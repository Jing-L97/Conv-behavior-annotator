#!/bin/bash

# ── positional arguments ──────────────────────────────────────────────────────
REWARD=$1
REWARD_SEED=$2
FINETUNE_SEED=$3
EXP=$4
EXP_SETTING=$5
DATA_SIZE=$6
PRETRAIN_SEED=$7

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/lustre/fsn1/projects/rech/eqb/uye44va/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
REWARD_PATH=$MODEL_ROOT/reward/$REWARD_SEED/$REWARD 
if [[ "$REWARD" == "topline" ]]; then
    REWARD_PATH=$MODEL_ROOT/reward/$REWARD
fi

echo "Reward model path: $REWARD_PATH"
echo "Policy model path: $MODEL_ROOT/lm/lightning_logs/$DATA_SIZE/$PRETRAIN_SEED/ckpt_huggingface_best/"
echo "output dir: $MODEL_ROOT/ppo/$EXP_SETTING/$PRETRAIN_SEED/$FINETUNE_SEED"
echo 'export LD_LIBRARY_PATH=/lustre/fsn1/projects/rech/eqb/uye44va/conda_envs/feedback/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

export LD_LIBRARY_PATH=/lustre/fsn1/projects/rech/eqb/uye44va/conda_envs/feedback/lib:$LD_LIBRARY_PATH


python -u $SCRIPT_ROOT/train/train_ppo.py \
    --policy_model $MODEL_ROOT/lm/lightning_logs/$DATA_SIZE/$PRETRAIN_SEED/ckpt_huggingface_best/ \
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
    --output_dir $MODEL_ROOT/ppo/$EXP_SETTING/$PRETRAIN_SEED/$FINETUNE_SEED \
    --wandb_dir $ROOT \
    --skip_existing \
    --seed $FINETUNE_SEED    
    
    