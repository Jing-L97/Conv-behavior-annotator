#!/bin/bash
#SBATCH --job-name=reward2
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/reward/2_%A_%a.log
#SBATCH --array=0-17
# ── core experiment properties ────────────────────────────────────────────────
COL_NAMES=(
    "is_cr"
    "is_acknowledgement"
    "align_lexical_unigram"
    "align_lexical_bigram"
    "align_syntactic"
    "align_semantic"
    "continuous_align_lexical_unigram"
    "continuous_align_lexical_bigram"
    "continuous_align_syntactic"
    "continuous_align_semantic"
    "sent_engagement"
    "sent_negativity"
    "sent_supportiveness"
    "sent_warmth"
    "sent_approval"
    "sent_caring"
    "sent_curiosity"
    "topline"
)
SEEDS=(2)
# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
WORKSPACE=$ROOT/"Conv-behavior-annotator/experiments/oberon/script/train"
cd $WORKSPACE
# ── array index validation ────────────────────────────────────────────────────
TOTAL_COMBINATIONS=$(( ${#COL_NAMES[@]} * ${#SEEDS[@]} ))
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi
# ── index resolution ──────────────────────────────────────────────────────────
COL_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
COL_NAME="${COL_NAMES[$COL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
# ── logging ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"
echo "  Column : $COL_NAME"
echo "  Seed   : $SEED"
echo "========================================================"
# ── launch ────────────────────────────────────────────────────────────────────
bash ./train_reward.sh $COL_NAME $SEED