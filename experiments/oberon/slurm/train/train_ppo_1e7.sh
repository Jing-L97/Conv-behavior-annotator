#!/bin/bash
#SBATCH --job-name=ppo_1e7
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/ppo/1e7_%A_%a.log
#SBATCH --array=0-1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# ── core experiment properties ────────────────────────────────────────────────

DATA_SIZES=("1e7")

PRETRAIN_SEEDS=(1 2)

REWARDS=(
    "sent_warmth"
)

FINETUNE_SEEDS=(3)


# ── dimension sizes ───────────────────────────────────────────────────────────
N_DATA=${#DATA_SIZES[@]}          # 3
N_PRETRAIN=${#PRETRAIN_SEEDS[@]}  # 2
N_REWARDS=${#REWARDS[@]}          # 18
N_FINETUNE=${#FINETUNE_SEEDS[@]}  # 4

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
WORKSPACE="${ROOT}/Conv-behavior-annotator/experiments/oberon/script/train"
cd "$WORKSPACE" || { echo "ERROR: Cannot cd to $WORKSPACE"; exit 1; }

# ── array index validation ────────────────────────────────────────────────────
# Layout: DATA_SIZE (outermost) → PRETRAIN_SEED → REWARD → FINETUNE_SEED (innermost)
TOTAL_COMBINATIONS=$(( N_DATA * N_PRETRAIN * N_REWARDS * N_FINETUNE ))

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) >= TOTAL_COMBINATIONS ($TOTAL_COMBINATIONS)"
    exit 1
fi

# ── index resolution ──────────────────────────────────────────────────────────
FINETUNE_IDX=$(( SLURM_ARRAY_TASK_ID % N_FINETUNE ))
REWARD_IDX=$(( (SLURM_ARRAY_TASK_ID / N_FINETUNE) % N_REWARDS ))
PRETRAIN_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_FINETUNE * N_REWARDS)) % N_PRETRAIN ))
DATA_IDX=$(( SLURM_ARRAY_TASK_ID / (N_FINETUNE * N_REWARDS * N_PRETRAIN) ))

DATA_SIZE="${DATA_SIZES[$DATA_IDX]}"
PRETRAIN_SEED="${PRETRAIN_SEEDS[$PRETRAIN_IDX]}"
REWARD="${REWARDS[$REWARD_IDX]}"
FINETUNE_SEED="${FINETUNE_SEEDS[$FINETUNE_IDX]}"

# ── reward seed override ──────────────────────────────────────────────────────
REWARD_SEED="$FINETUNE_SEED"
if [[ "$REWARD" == "topline" ]]; then
    REWARD_SEED=3
fi

# ── experiment tags ───────────────────────────────────────────────────────────
EXP_SETTING="${DATA_SIZE}_entropy_001_lm_loss_001_target_6"
EXP="${DATA_SIZE}_reward_seed_${REWARD_SEED}_entropy_001_lm_loss_001_target_6"

# ── logging ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Task ID      : $SLURM_ARRAY_TASK_ID / $TOTAL_COMBINATIONS"
echo "  Data size    : $DATA_SIZE  (idx=$DATA_IDX)"
echo "  Pretrain seed: $PRETRAIN_SEED  (idx=$PRETRAIN_IDX)"
echo "  Reward       : $REWARD  (idx=$REWARD_IDX)"
echo "  Finetune seed: $FINETUNE_SEED  (idx=$FINETUNE_IDX)"
echo "  Reward seed  : $REWARD_SEED"
echo "  EXP tag      : $EXP"
echo "  EXP setting  : $EXP_SETTING"
echo "========================================================"

# ── launch ────────────────────────────────────────────────────────────────────
bash ./train_ppo.sh \
    "$REWARD"        \
    "$REWARD_SEED"   \
    "$FINETUNE_SEED" \
    "$EXP"           \
    "$EXP_SETTING"   \
    "$DATA_SIZE"     \
    "$PRETRAIN_SEED"