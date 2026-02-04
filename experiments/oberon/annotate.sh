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

# Run the script with the appropriate configuration
python $SCRIPT_ROOT/annotate.py --annotate_sentiment --annotate_alignment --exclude_interjections

python $SCRIPT_ROOT/annotate.py --annotate_alignment --exclude_stopwords

python $SCRIPT_ROOT/annotate.py --annotate_alignment 