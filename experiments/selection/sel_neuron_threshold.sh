#!/bin/bash
#SBATCH --job-name=sel_single
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/selection/sel_single_%a.log
#SBATCH --array=0-41  # adjust depending on total combinations

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/selection"

# Define the input arrays
EFFECTS=("boost")
TOP_NS=(-1)
MODELS=(
    "EleutherAI/pythia-1B-deduped"
    "EleutherAI/pythia-1.4B-deduped"
    "EleutherAI/pythia-2.8B-deduped"
    "gpt2"
    "gpt2-medium"
    "gpt2-large"
    "gpt2-xl"
)
INTERVALS=(15 20 25)
SEL_FREQS=("longtail" "common")

# Total combinations
TOTAL_COMBINATIONS=$((${#EFFECTS[@]} * ${#TOP_NS[@]} * ${#MODELS[@]} * ${#INTERVALS[@]} * ${#SEL_FREQS[@]}))

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate indices
EFFECT_IDX=$(( SLURM_ARRAY_TASK_ID / (${#TOP_NS[@]} * ${#MODELS[@]} * ${#INTERVALS[@]} * ${#SEL_FREQS[@]}) ))
TOP_N_IDX=$(( (SLURM_ARRAY_TASK_ID / (${#MODELS[@]} * ${#INTERVALS[@]} * ${#SEL_FREQS[@]})) % ${#TOP_NS[@]} ))
MODEL_IDX=$(( (SLURM_ARRAY_TASK_ID / (${#INTERVALS[@]} * ${#SEL_FREQS[@]})) % ${#MODELS[@]} ))
INTERVAL_IDX=$(( (SLURM_ARRAY_TASK_ID / ${#SEL_FREQS[@]}) % ${#INTERVALS[@]} ))
SEL_FREQ_IDX=$(( SLURM_ARRAY_TASK_ID % ${#SEL_FREQS[@]} ))

# Assign values
EFFECT="${EFFECTS[$EFFECT_IDX]}"
TOP_N="${TOP_NS[$TOP_N_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"
INTERVAL="${INTERVALS[$INTERVAL_IDX]}"
SEL_FREQ="${SEL_FREQS[$SEL_FREQ_IDX]}"

# Log info
echo "Processing combination:"
echo " Effect: $EFFECT"
echo " Top N: $TOP_N"
echo " Model: $MODEL"
echo " Interval: $INTERVAL"
echo " Sel Freq: $SEL_FREQ"

# Run the analysis
python $SCRIPT_ROOT/sel_neuron.py \
    -m "$MODEL" \
    --effect "$EFFECT" \
    --top_n "$TOP_N" \
    --sel_freq "$SEL_FREQ" \
    --interval "$INTERVAL"
