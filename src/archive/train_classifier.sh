#!/bin/bash
#SBATCH --job-name=annotate_all
#SBATCH --export=ALL
#SBATCH --partition=gpu-p1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/Feedback/logs/annot/annotate.log


# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Feedback/Conv-behavior-annotator/src/scripts"
MODEL_ROOT="/scratch2/jliu/Feedback/Conv-behavior-annotator/src/scripts"
# Run the script with the appropriate configuration

python $SCRIPT_ROOT/train_classifier.py \
    --data_path annotated/conversations_min_age_10_no_intjs.csv \
    --target_column is_cr \
    --model_name_or_path microsoft/deberta-v3-xsmall \
    --output_dir models/cr_classifier