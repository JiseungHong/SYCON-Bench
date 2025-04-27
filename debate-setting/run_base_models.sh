#!/bin/bash

# run_qwen_models.sh
# Script to run the mt-prompt.py script with the three Qwen base models

# Set common parameters
BATCH_SIZE=1
NUM_RESPONSES=5
OUTPUT_DIR="output"

# Define models to run
MODELS=(
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-72B"
)

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print start message
echo "Starting processing of all Qwen base models at $(date)"

# Process each model
for MODEL in "${MODELS[@]}"; do
    MODEL_ID=$(echo "$MODEL" | cut -d'/' -f2)
    
    echo "=========================================================="
    echo "Starting processing of model: $MODEL"
    echo "Started at: $(date)"
    echo "=========================================================="
    
    # Run the Python script for this model
    python mt-prompt-base-models.py "$MODEL"
    
    # Check if the script execution was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed model: $MODEL"
    else
        echo "Error processing model: $MODEL"
    fi
    
    echo "Finished at: $(date)"
    echo "=========================================================="
    
    # Add a short pause between model runs
    sleep 5
done

echo "All models have been processed."
echo "Results can be found in the following directories:"

# Print paths to result files
for MODEL in "${MODELS[@]}"; do
    MODEL_ID=$(echo "$MODEL" | cut -d'/' -f2)
    echo "- ${OUTPUT_DIR}/${MODEL_ID}/prompt0.csv"
done