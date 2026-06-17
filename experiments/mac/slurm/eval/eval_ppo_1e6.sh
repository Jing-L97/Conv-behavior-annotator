#!/bin/bash
#SBATCH --job-name=eval_1e6
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/eval_align/1e6_%A_%a.log
#SBATCH --array=0-131%4

# ── core experiment properties ────────────────────────────────────────────────

DATA_SIZES=("1e6")

PRETRAIN_SEEDS=(1 2)

REWARDS=(
    "is_cr"
    "is_acknowledgement"
    "align_lexical_unigram"
    "align_lexical_bigram"
    "align_syntactic"
    "continuous_align_lexical_unigram"
    "continuous_align_lexical_bigram"
    "continuous_align_syntactic"
    "continuous_align_semantic"
    "align_semantic"
    "sent_warmth"

)

FINETUNE_SEEDS=(3 1024)

GEN_SEEDS=(1 2 3)


# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
WORKSPACE=$ROOT/"Conv-behavior-annotator/experiments/oberon/script/eval"
cd $WORKSPACE

# ── array index validation ────────────────────────────────────────────────────
# Layout: DATA_SIZE → FINETUNE_SEED → PRETRAIN_SEED → REWARD → GEN_SEED (inner)
TOTAL_COMBINATIONS=$(( ${#DATA_SIZES[@]} * ${#FINETUNE_SEEDS[@]} * ${#PRETRAIN_SEEDS[@]} * ${#REWARDS[@]} * ${#GEN_SEEDS[@]} ))

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# ── index resolution ──────────────────────────────────────────────────────────
INNER_1=$(( ${#FINETUNE_SEEDS[@]} * ${#PRETRAIN_SEEDS[@]} * ${#REWARDS[@]} * ${#GEN_SEEDS[@]} ))
INNER_2=$(( ${#PRETRAIN_SEEDS[@]} * ${#REWARDS[@]} * ${#GEN_SEEDS[@]} ))
INNER_3=$(( ${#REWARDS[@]} * ${#GEN_SEEDS[@]} ))
INNER_4=$(( ${#GEN_SEEDS[@]} ))

DATA_IDX=$((         SLURM_ARRAY_TASK_ID / INNER_1 ))
FINETUNE_IDX=$((   ( SLURM_ARRAY_TASK_ID % INNER_1 ) / INNER_2 ))
PRETRAIN_IDX=$((   ( SLURM_ARRAY_TASK_ID % INNER_2 ) / INNER_3 ))
REWARD_IDX=$((     ( SLURM_ARRAY_TASK_ID % INNER_3 ) / INNER_4 ))
GEN_SEED_IDX=$((     SLURM_ARRAY_TASK_ID % INNER_4 ))

DATA_SIZE="${DATA_SIZES[$DATA_IDX]}"
FINETUNE_SEED="${FINETUNE_SEEDS[$FINETUNE_IDX]}"
PRETRAIN_SEED="${PRETRAIN_SEEDS[$PRETRAIN_IDX]}"
REWARD="${REWARDS[$REWARD_IDX]}"
GEN_SEED="${GEN_SEEDS[$GEN_SEED_IDX]}"

REWARD_SEED=$FINETUNE_SEED
if [[ "$REWARD" == "topline" ]]; then
    REWARD_SEED=3
fi

EXP_SETTING="${DATA_SIZE}_entropy_001_lm_loss_001_target_6"
EXP="${DATA_SIZE}_reward_seed_${REWARD_SEED}_entropy_001_lm_loss_001_target_6"

# ── logging ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"
echo "  Data size     : $DATA_SIZE"
echo "  Finetune seed : $FINETUNE_SEED"
echo "  Pretrain seed : $PRETRAIN_SEED"
echo "  Reward        : $REWARD"
echo "  Gen seed      : $GEN_SEED"
echo "  EXP tag       : $EXP"
echo "========================================================"

# ── launch ────────────────────────────────────────────────────────────────────
bash ./eval_ppo.sh "$REWARD" "$FINETUNE_SEED" "$EXP" "$EXP_SETTING" "$GEN_SEED" "$PRETRAIN_SEED"