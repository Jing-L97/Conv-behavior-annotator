#!/bin/bash
#SBATCH --job-name=train_reward
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/rlhf/reward_%A_%a.log
#SBATCH --array=0-5

# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"

# Define column names as an array
COL_NAMES=("sent_approval" "sent_caring" "sent_curiosity" "align_lexical_unigram" "align_lexical_bigram" "align_syntactic")

# Get the column name for this array task
COL_NAME=${COL_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Training reward model for column: $COL_NAME"

python $SCRIPT_ROOT/train_reward.py \
    --data_path $DATA_ROOT/annotated/conversations_min_age_10_all.csv \
    --model_name_or_path microsoft/deberta-v3-xsmall \
    --output_dir $MODEL_ROOT/reward/$COL_NAME \
    --reward_column_name $COL_NAME \
    --wandb_dir $ROOT \
    --resume_from_checkpoint
