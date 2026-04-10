#!/bin/bash
#SBATCH --job-name=ppo_align
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/ppo/align_1e7_%A_%a.log
#SBATCH --array=0-9

# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
EXP="1e7_reward_seed_3_entropy_001_lm_loss_001_target_6"
LM="uu5rtja8"

# Define column names as an array
REWARDS=("is_cr" "is_acknowledgement" "align_lexical_unigram" "align_lexical_bigram" "align_syntactic" "align_semantic" "continuous_align_lexical_unigram" "continuous_align_lexical_bigram" "continuous_align_syntactic" "continuous_align_semantic")
#SEEDS=(123 999 1024)
SEEDS=(3)

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


# Run the script with the appropriate configuration
python -u $SCRIPT_ROOT/train/train_ppo.py \
    --policy_model $MODEL_ROOT/lm/lightning_logs/$LM/ckpt_huggingface_best/ \
    --value_model $MODEL_ROOT/reward/$SEED/$REWARD \
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
    --output_dir $MODEL_ROOT/ppo/$REWARD_$EXP/$SEED \
    --wandb_dir $ROOT \
    --skip_existing \
    --seed $SEED


