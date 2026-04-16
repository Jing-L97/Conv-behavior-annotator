#!/bin/bash
#SBATCH --job-name=eval_baseline_1e5
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/baselin_1e5e_%A_%a.log
#SBATCH --array=0-1


# ── core experiment properties ────────────────────────────────────────────────
# DATA_SIZES=("1e5" "1e6" "1e7")
# LMS=("967ufsfk" "he3nnzld" "uu5rtja8")

DATA_SIZES=("1e7")
LMS=("uu5rtja8")

SEEDS=(3)

ROOT="/scratch2/jliu/Feedback"
WORKSPACE="$ROOT/Conv-behavior-annotator/experiments/oberon/script/eval"
cd "$WORKSPACE"

TOTAL_COMBINATIONS=$(( ${#DATA_SIZES[@]} * ${#SEEDS[@]} ))

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

DATA_IDX=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))

DATA_SIZE="${DATA_SIZES[$DATA_IDX]}"
LM="${LMS[$DATA_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

echo "========================================================"
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"
echo "  Data size : $DATA_SIZE  (LM: $LM)"
echo "  Seed      : $SEED"
echo "========================================================"

bash ./eval_grammar_baseline.sh "$LM" "$SEED" "$DATA_SIZE"
