#!/bin/bash
#SBATCH --job-name=ppo_align
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/rlhf/ppo_eval.log
#SBATCH --array=0-1

# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
OUT_ROOT=$ROOT/"results"
EXP="1e6_reward_seed_3_entropy_001_lm_loss_001_target_6"

# Define column names as an array
REWARDS=(“cr” "is_acknowledgement" "align_lexical_unigram" "align_lexical_bigram" "align_syntactic" "align_semantic" "sent_engagement" "sent_negativity" "sent_supportiveness" "sent_warmth" "sent_approval" "sent_caring" "sent_curiosity")

# Get the column name for this array task
REWARD=${REWARDS[$SLURM_ARRAY_TASK_ID]}

REWARD=cr
PPO_MODEL=$MODEL_ROOT/ppo/$EXP/${REWARD}_$EXP/best_reward
OUTPUT_DIR=$ROOT/results/ppo/$EXP/${REWARD}_$EXP

python -u $SCRIPT_ROOT/eval/eval_gen.py \
    --model_paths $PPO_MODEL \
    --eval_model_path $MODEL_ROOT/grammar_eval/version_19 \
    --word_info_path $DATA_ROOT/evaluation_data/gen/word_info.csv \
    --func_info_path $DATA_ROOT/evaluation_data/gen/func_info.csv \
    --output_utts_csv $OUTPUT_DIR/utt.csv \
    --output_csv $OUTPUT_DIR/output.csv \
    --batch_size 64 --num_batches 2 

