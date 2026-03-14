#!/bin/bash
#SBATCH --job-name=reward_sent
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/reward/sent_%A_%a.log
#SBATCH --array=0-23

# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"

# Define column names as an array
COL_NAMES=("sent_engagement" "sent_negativity" "sent_negativity_reverse" "sent_supportiveness" "sent_warmth" "sent_approval" "sent_caring" "sent_curiosity")
# Get the column name for this array task
COL_NAME=${COL_NAMES[$SLURM_ARRAY_TASK_ID]}

SEEDS=(123 999 1024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Training reward model for column: $COL_NAME"

python $SCRIPT_ROOT/train/train_reward.py \
    --data_path $DATA_ROOT/annotated/conversations_min_age_10.csv \
    --model_name_or_path microsoft/deberta-v3-xsmall \
    --output_dir $MODEL_ROOT/reward/$SEED/$COL_NAME \
    --reward_column_name $COL_NAME \
    --wandb_dir $ROOT \
    --seed $SEED
