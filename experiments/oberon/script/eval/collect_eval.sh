#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/eval/eval_gen_vanilla.log

# Script and config paths
ROOT="/scratch2/jliu/Feedback"
SCRIPT_ROOT=$ROOT/"Conv-behavior-annotator/src/scripts"


python -u $SCRIPT_ROOT/eval/collect_eval.py --collect_utterances