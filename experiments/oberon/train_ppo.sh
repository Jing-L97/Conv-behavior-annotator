#!/bin/bash
#SBATCH --job-name=ppo_cr
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/rlhf/ppo_cr.log


# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Feedback/Conv-behavior-annotator/src/scripts"
MODEL_ROOT="/scratch2/jliu/Feedback/models"
DATA_ROOT="/scratch2/jliu/Feedback/datasets"


# Run the script with the appropriate configuration
python -u $SCRIPT_ROOT/train_ppo.py \
    --policy_model $MODEL_ROOT/lm/lightning_logs/he3nnzld/ckpt_huggingface_best/ \
    --value_model $MODEL_ROOT/reward/cr/checkpoint-2650/ \
    --steps 6000 \
    --target 6 \
    --lm_data_path $DATA_ROOT/raw/caregiver_utterances_train_1000000.0_words.txt \
    --lm_val_data_path $DATA_ROOT/raw/caregiver_utterances_val_1000000.0_words.txt \
    --grammar_eval_model_path $MODEL_ROOT/grammar_eval/version_19 \
    --entropy_reg_coef 0.001 \
    --length_reward_coef 0 \
    --lm_loss_coef 0.001 \
    --exp_name 1e6_reward_topline_seed_3_entropy_001_lm_loss_001_target_6 \
    --seed 3


