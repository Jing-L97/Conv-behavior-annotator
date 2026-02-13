#!/bin/bash
#SBATCH --job-name=train_reward
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/rlhf/reward_%A_%a.log
#SBATCH --array=0-8

# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Feedback/Conv-behavior-annotator/src/scripts"
MODEL_ROOT="/scratch2/jliu/Feedback/models"
DATA_ROOT="/scratch2/jliu/Feedback/datasets"

# Define column names as an array
COL_NAMES=("is_acknowledgement" "sent_approval" "sent_caring" "sent_curiosity" "sent_engagement" "sent_negativity" "sent_supportiveness" "sent_warmth" "align_semantic")

# Get the column name for this array task
COL_NAME=${COL_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Training reward model for column: $COL_NAME"

# Run the script with the appropriate configuration
python $SCRIPT_ROOT/train_reward.py \
    --data_path $DATA_ROOT/annotated/conversations_min_age_10_no_intjs.csv \
    --model_name_or_path microsoft/deberta-v3-xsmall \
    --output_dir $MODEL_ROOT/reward/$COL_NAME \
    --reward_column_name $COL_NAME