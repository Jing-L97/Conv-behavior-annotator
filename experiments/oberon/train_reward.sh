#!/bin/bash
#SBATCH --job-name=train_reward
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/rlhf/reward_cr.log


# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Feedback/Conv-behavior-annotator/src/scripts"
MODEL_ROOT="/scratch2/jliu/Feedback/models"
DATA_ROOT="/scratch2/jliu/Feedback/datasets"
# Run the script with the appropriate configuration

python $SCRIPT_ROOT/train_reward.py \
    --data_path $DATA_ROOT/annotated/conversations_min_age_10_no_intjs.csv \
    --model_name_or_path microsoft/deberta-v3-xsmall \
    --output_dir $MODEL_ROOT/reward/cr

