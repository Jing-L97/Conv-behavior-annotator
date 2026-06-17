#!/bin/bash
#SBATCH --job-name=eval_grammar
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/garmmar_%A_%a.log
#SBATCH --array=0-5

# ── core experiment properties ────────────────────────────────────────────────
DATA_SIZES=("1e5" "1e6" "1e7")
SEEDS=(1 2)


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
SEED="${SEEDS[$SEED_IDX]}"

echo "========================================================"
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"
echo "  Data size : $DATA_SIZE"
echo "  Seed      : $SEED"
echo "========================================================"

bash ./eval_grammar_baseline.sh "$SEED" "$DATA_SIZE"