#!/bin/bash
#SBATCH --job-name=ppo_1e5_seed2
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/ppo/1e5_seed2_%A_%a.log
#SBATCH --array=0-16%6

# ── core experiment properties ────────────────────────────────────────────────
DATA_SIZES=("1e5")
LMS=("95f8k8zc")
REWARDS=(
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
    )
SEEDS=(2)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
WORKSPACE=$ROOT/"Conv-behavior-annotator/experiments/oberon/script/train"
cd $WORKSPACE

# ── array index validation ────────────────────────────────────────────────────
TOTAL_COMBINATIONS=$(( ${#DATA_SIZES[@]} * ${#REWARDS[@]} * ${#SEEDS[@]} ))
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# ── index resolution ──────────────────────────────────────────────────────────
# Layout: DATA_SIZE (outer) → REWARD (middle) → SEED (inner)
DATA_IDX=$(( SLURM_ARRAY_TASK_ID / (${#REWARDS[@]} * ${#SEEDS[@]}) ))
REWARD_IDX=$(( (SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}) % ${#REWARDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

DATA_SIZE="${DATA_SIZES[$DATA_IDX]}"
LM="${LMS[$DATA_IDX]}"
REWARD="${REWARDS[$REWARD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
REWARD_SEED=$SEED
if [[ "$REWARD" == "topline" ]]; then
    REWARD_SEED=3 
fi

EXP_SETTING="${DATA_SIZE}_entropy_001_lm_loss_001_target_6"
EXP="${DATA_SIZE}_reward_seed_${REWARD_SEED}_entropy_001_lm_loss_001_target_6"


# ── logging ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"
echo "  Data size : $DATA_SIZE  (LM: $LM)"
echo "  Reward    : $REWARD"
echo "  PPO Seed      : $SEED"
echo "  Reward Seed      : $REWARD_SEED"
echo "  EXP tag   : $EXP"
echo "========================================================"

# ── launch ────────────────────────────────────────────────────────────────────
bash ./train_ppo.sh $REWARD $REWARD_SEED $SEED $EXP $EXP_SETTING $LM