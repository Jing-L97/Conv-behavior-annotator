#!/bin/bash
#SBATCH --job-name=eval_baseline
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/baseline/%A_%a.log
#SBATCH --array=0-11
# ── core experiment properties ────────────────────────────────────────────────
DATA_SIZES=("1e5" "1e6" "1e7")
LMS=("967ufsfk" "he3nnzld" "uu5rtja8")
SEEDS=(3)
GEN_SEEDS=(1024 123 3 999)
ROOT="/scratch2/jliu/Feedback"
WORKSPACE="$ROOT/Conv-behavior-annotator/experiments/oberon/script/eval"
cd "$WORKSPACE"
# ── array index validation ────────────────────────────────────────────────────
# Layout: DATA_SIZE (outer) → SEED → GEN_SEED (inner)
TOTAL_COMBINATIONS=$(( ${#DATA_SIZES[@]} * ${#SEEDS[@]} * ${#GEN_SEEDS[@]} ))
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi
# ── index resolution ──────────────────────────────────────────────────────────
DATA_IDX=$(( SLURM_ARRAY_TASK_ID / (${#SEEDS[@]} * ${#GEN_SEEDS[@]}) ))
SEED_IDX=$(( (SLURM_ARRAY_TASK_ID / ${#GEN_SEEDS[@]}) % ${#SEEDS[@]} ))
GEN_SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#GEN_SEEDS[@]} ))
DATA_SIZE="${DATA_SIZES[$DATA_IDX]}"
LM="${LMS[$DATA_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
GEN_SEED="${GEN_SEEDS[$GEN_SEED_IDX]}"
# ── logging ───────────────────────────────────────────────────────────────────
echo "========================================================"
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"
echo "  Data size : $DATA_SIZE  (LM: $LM)"
echo "  Seed      : $SEED"
echo "  Gen seed  : $GEN_SEED"
echo "========================================================"
# ── launch ────────────────────────────────────────────────────────────────────
bash ./eval_baseline.sh "$LM" "$SEED" "$DATA_SIZE" "$GEN_SEED"


# bash ./eval_grammar_baseline.sh "$LM" "$SEED" "$DATA_SIZE" 