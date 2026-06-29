#!/bin/bash
#SBATCH --job-name=eval_grammar_type
#SBATCH --export=ALL
#SBATCH --partition=erc-dupoux
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/eval_grammar_type/%A_%a.log
#SBATCH --array=0-1

# ── core experiment properties ────────────────────────────────────────────────
JUDGE_MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-4B")

# pick model for this array task
TASK_ID=${SLURM_ARRAY_TASK_ID}
JUDGE_MODEL=${JUDGE_MODELS[$TASK_ID]}

echo "Running task ${TASK_ID} with model: ${JUDGE_MODEL}"

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT="/scratch2/jliu/Feedback"
WORKSPACE="${ROOT}/Conv-behavior-annotator/experiments/oberon/script/eval"

cd "$WORKSPACE" || exit 1

# ── launch ────────────────────────────────────────────────────────────────────
bash ./eval_grammar_type.sh "$JUDGE_MODEL"