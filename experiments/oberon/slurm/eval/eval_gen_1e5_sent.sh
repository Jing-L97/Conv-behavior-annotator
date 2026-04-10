#!/bin/bash
#SBATCH --job-name=eval_gen_1e5_sent
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/eval_gen_1e5_seed_sent_%A_%a.log
#SBATCH --array=0-20

# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
OUT_ROOT=$ROOT/"results"
EXP="1e5_reward_seed_3_entropy_001_lm_loss_001_target_6"

# Define column names as an array
REWARDS=(
    "sent_engagement" 
    "sent_negativity" 
    "sent_supportiveness" 
    "sent_warmth" 
    "sent_approval" 
    "sent_caring" 
    "sent_curiosity"
)
SEEDS=(123 999 1024)

#SEEDS=(3)


# Calculate total combinations for validation
TOTAL_COMBINATIONS=$(( ${#REWARDS[@]} * ${#SEEDS[@]} ))
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate indices using integer division and modulo
REWARD_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

# Get actual values
REWARD="${REWARDS[$REWARD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# Log which combination is being processed
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS:"
echo "  Reward : $REWARD"
echo "  Seed   : $SEED"


PPO_MODEL=$MODEL_ROOT/ppo/$EXP/$SEED/${REWARD}_$EXP/best_reward
OUTPUT_DIR=$ROOT/results/ppo/$EXP/$SEED/${REWARD}_$EXP


python -u $SCRIPT_ROOT/eval/eval_gen.py \
    --model_paths $PPO_MODEL \
    --eval_model_path $MODEL_ROOT/grammar_eval/version_19 \
    --word_info_path $DATA_ROOT/evaluation_data/gen/word_info.csv \
    --func_info_path $DATA_ROOT/evaluation_data/gen/func_info.csv \
    --output_utts_csv $OUTPUT_DIR/utt.csv \
    --output_csv $OUTPUT_DIR/result.csv \
    --batch_size 50 \
    --num_batches 200 

