#!/bin/bash
#SBATCH --job-name=modularity
#SBATCH --export=ALL
#SBATCH --partition=erc-cristia 
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/analysis/modularity_%a.log
#SBATCH --array=0-4  # Matches number of MODELS

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/analysis"

# Define parameter arrays
MODELS=(
  "gpt2-large" 
  "gpt2-xl"
  "EleutherAI/pythia-1B-deduped"
  "EleutherAI/pythia-1.4B-deduped"
  "EleutherAI/pythia-2.8B-deduped"
)
INTERVAL=15

# Map SLURM array index to model
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Running geometry analysis for model: $MODEL (interval=$INTERVAL)"

# Run the analysis script
python "$SCRIPT_ROOT/activation_modularity.py" \
  -m "$MODEL" \
  --interval "$INTERVAL" \
  --regime \
  --resume
