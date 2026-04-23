#!/bin/bash
#SBATCH --job-name=eval_childes
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/childes_%A_%a.log
#SBATCH --array=0-1

# ── core experiment properties ────────────────────────────────────────────────
COLS=("utt_transcript_clean" "response_transcript_clean")
COL="${COLS[$SLURM_ARRAY_TASK_ID]}"   # pick the column for this array task

ROOT="/scratch2/jliu/Feedback"
WORKSPACE="$ROOT/Conv-behavior-annotator/experiments/oberon/script/eval"
cd "$WORKSPACE"

bash ./eval_childes.sh "$COL"