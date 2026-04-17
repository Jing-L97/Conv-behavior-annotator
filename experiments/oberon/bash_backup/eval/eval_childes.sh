#!/bin/bash
#SBATCH --job-name=eval_gen_1e5_align
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/eval_gen_1e5_seed_align_%A_%a.log
#SBATCH --array=0-29

# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
OUT_ROOT=$ROOT/"results/childes"
COL="utt_transcript_clean"


python -u $SCRIPT_ROOT/eval/eval_childes.py \
    --input_csv $DATA_ROOT/raw/conversations_min_age_10.csv \
    --utt_column $COL \
    --eval_model_path $MODEL_ROOT/grammar_eval/version_19 \
    --word_info_path $DATA_ROOT/evaluation_data/gen/word_info.csv \
    --func_info_path $DATA_ROOT/evaluation_data/gen/func_info.csv \
    --output_utts_csv $OUT_ROOT/$COL/utt.csv \
    --output_csv $OUT_ROOT/$COL/result.csv \
    --debug

