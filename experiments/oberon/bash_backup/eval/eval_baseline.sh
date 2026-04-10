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
MODEL_ROOT=$ROOT/"models"
DATA_ROOT=$ROOT/"datasets"
OUT_ROOT=$ROOT/"results"
EXP="1e6_reward_seed_3_entropy_001_lm_loss_001_target_6"


PPO_MODEL=$MODEL_ROOT/lm/lightning_logs/he3nnzld/ckpt_huggingface_best
OUTPUT_DIR=$ROOT/results/baseline

python -u $SCRIPT_ROOT/eval/eval_gen.py \
    --model_paths $PPO_MODEL \
    --eval_model_path $MODEL_ROOT/grammar_eval/version_19 \
    --word_info_path $DATA_ROOT/evaluation_data/gen/word_info.csv \
    --func_info_path $DATA_ROOT/evaluation_data/gen/func_info.csv \
    --output_utts_csv $OUTPUT_DIR/utt.csv \
    --output_csv $OUTPUT_DIR/result.csv \
    --batch_size 50 \
    --num_batches 10 \
    --baseline

