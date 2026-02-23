#!/bin/bash
#SBATCH --job-name=ppo_cr
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/rlhf/ppo_cr.log


# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
EXP="1e6_reward_seed_3_entropy_001_lm_loss_001_target_6"

# Get the column name for this array task
REWARD="cr"

# Run the script with the appropriate configuration
python -u $SCRIPT_ROOT/train_ppo.py \
    --policy_model $MODEL_ROOT/lm/lightning_logs/he3nnzld/ckpt_huggingface_best/ \
    --value_model $MODEL_ROOT/reward/$REWARD \
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
    --output_dir $MODEL_ROOT/ppo/$REWARD_$EXP \
    --wandb_dir $ROOT \
    --seed 3


