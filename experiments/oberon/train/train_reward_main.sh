#!/bin/bash
#SBATCH --job-name=reward_main
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/reward/main_%A_%a.log
#SBATCH --array=0-17

# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"

# Define parameter arrays
COL_NAMES=(
    "is_cr"
    "is_acknowledgement"
    "align_lexical_unigram"
    "align_lexical_bigram"
    "align_syntactic"
    "align_semantic"
)

SEEDS=(
    123
    999
    1024
)

# Calculate total combinations for validation
TOTAL_COMBINATIONS=$(( ${#COL_NAMES[@]} * ${#SEEDS[@]} ))
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate indices using integer division and modulo
COL_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

# Get actual values
COL_NAME="${COL_NAMES[$COL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# Log which combination is being processed
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS:"
echo "  Column : $COL_NAME"
echo "  Seed   : $SEED"

python $SCRIPT_ROOT/train/train_reward.py \
    --data_path $DATA_ROOT/annotated/conversations_min_age_10.csv \
    --model_name_or_path microsoft/deberta-v3-xsmall \
    --output_dir $MODEL_ROOT/reward/$SEED/$COL_NAME \
    --reward_column_name $COL_NAME \
    --wandb_dir $ROOT \
    --skip_existing \
    --seed $SEED