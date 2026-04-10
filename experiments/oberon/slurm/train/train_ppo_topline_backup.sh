#!/bin/bash
#SBATCH --job-name=ppo_topline
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/ppo/topline_1e6_%A_%a.log
#SBATCH --array=0-1

# ── core experiment properties ────────────────────────────────────────────────
DATA_SIZE="1e6"
LM="he3nnzld"

DATA_SIZE="1e7"
LM="uu5rtja8"

DATA_SIZE="1e5"
LM="u967ufsfk"

EXP=$DATA_SIZE"_reward_seed_3_entropy_001_lm_loss_001_target_6"
REWARDS=("topline")
SEEDS=(3 123 999 1024)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
WORKSPACE=$ROOT/"Conv-behavior-annotator/experiments/oberon/script/train"
cd $WORKSPACE

# ── array index validation ────────────────────────────────────────────────────
TOTAL_COMBINATIONS=$(( ${#REWARDS[@]} * ${#SEEDS[@]} ))
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# ── index resolution ──────────────────────────────────────────────────────────
REWARD_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

REWARD="${REWARDS[$REWARD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# ── logging ───────────────────────────────────────────────────────────────────
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS:"
echo "  Reward : $REWARD"
echo "  Seed   : $SEED"

# ── launch ────────────────────────────────────────────────────────────────────
bash ./train_ppo.sh $REWARD $SEED $EXP $LM