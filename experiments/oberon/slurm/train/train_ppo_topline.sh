#!/bin/bash
#SBATCH --job-name=ppo_topline
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/ppo/topline_%A_%a.log
#SBATCH --array=0-11

# ── core experiment properties ────────────────────────────────────────────────
DATA_SIZES=("1e5"      "1e6"      "1e7")
LMS=(       "967ufsfk" "he3nnzld" "uu5rtja8")
REWARDS=("topline")
SEEDS=(3 123 999 1024)

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

EXP="${DATA_SIZE}_reward_seed_3_entropy_001_lm_loss_001_target_6"

# ── logging ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"
echo "  Data size : $DATA_SIZE  (LM: $LM)"
echo "  Reward    : $REWARD"
echo "  Seed      : $SEED"
echo "  EXP tag   : $EXP"
echo "========================================================"

# ── launch ────────────────────────────────────────────────────────────────────
bash ./train_ppo.sh $REWARD $SEED $EXP $LM