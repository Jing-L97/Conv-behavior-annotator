#!/bin/bash
#SBATCH --job-name=eval_sel_seeds_2
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/sel_seeds_2/%A_%a.log
#SBATCH --array=0-599%8
# ── core experiment properties ────────────────────────────────────────────────
DATA_SIZES=("1e5" "1e6" "1e7")
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
"sent_warmth"
)
SEEDS=(1024 123 3 999)
GEN_SEEDS=(0 1 2 42 7)
# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
WORKSPACE=$ROOT/"Conv-behavior-annotator/experiments/oberon/script/eval"
cd $WORKSPACE

# ── array index validation ────────────────────────────────────────────────────
TOTAL_COMBINATIONS=$(( ${#DATA_SIZES[@]} * ${#REWARDS[@]} * ${#SEEDS[@]} * ${#GEN_SEEDS[@]} ))
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi
# ── index resolution ──────────────────────────────────────────────────────────
# Layout: DATA_SIZE (outer) → REWARD → SEED → GEN_SEED (inner)
DATA_IDX=$(( SLURM_ARRAY_TASK_ID / (${#REWARDS[@]} * ${#SEEDS[@]} * ${#GEN_SEEDS[@]}) ))
REWARD_IDX=$(( (SLURM_ARRAY_TASK_ID / (${#SEEDS[@]} * ${#GEN_SEEDS[@]})) % ${#REWARDS[@]} ))
SEED_IDX=$(( (SLURM_ARRAY_TASK_ID / ${#GEN_SEEDS[@]}) % ${#SEEDS[@]} ))
GEN_SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#GEN_SEEDS[@]} ))
DATA_SIZE="${DATA_SIZES[$DATA_IDX]}"
REWARD="${REWARDS[$REWARD_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
GEN_SEED="${GEN_SEEDS[$GEN_SEED_IDX]}"
REWARD_SEED=$SEED
if [[ "$REWARD" == "topline" ]]; then
    REWARD_SEED=3
fi
EXP_SETTING="${DATA_SIZE}_entropy_001_lm_loss_001_target_6"
EXP="${DATA_SIZE}_reward_seed_${REWARD_SEED}_entropy_001_lm_loss_001_target_6"
# ── logging ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"
echo "  Data size : $DATA_SIZE"
echo "  Reward    : $REWARD"
echo "  Seed      : $SEED"
echo "  Gen seed  : $GEN_SEED"
echo "  EXP tag   : $EXP"
echo "========================================================"
# ── launch ────────────────────────────────────────────────────────────────────
bash ./eval_ppo.sh "$REWARD" "$SEED" "$EXP" "$EXP_SETTING" "$GEN_SEED"