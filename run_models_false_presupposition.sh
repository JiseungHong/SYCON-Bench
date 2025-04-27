#!/bin/bash

# run_models_false_presupposition.sh
# Script to run the mt-prompt-vllm.py script with multiple models in sequence

# Change to the false-presuppositions-setting directory
cd false-presuppositions-setting

# Define models to run in the specified order
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "google/gemma-2-9b-it"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
)

# Print start message
echo "========================================================"
echo "Starting processing of all models at $(date)"
echo "========================================================"

# Process each model in sequence
for MODEL in "${MODELS[@]}"; do
    MODEL_ID=$(echo "$MODEL" | cut -d'/' -f2)
    
    echo "========================================================"
    echo "Starting processing of model: $MODEL"
    echo "Started at: $(date)"
    echo "========================================================"
    
    # Run the Python script with vLLM for this model (using default parameters)
    python mt-prompt-vllm.py "$MODEL"
    
    # Check if the script execution was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed model: $MODEL"
    else
        echo "Error processing model: $MODEL"
    fi
    
    echo "Finished at: $(date)"
    echo "$MODEL is finished."
    echo "========================================================"
    
    # Add a short pause between model runs
    sleep 10
done

echo "All models have been processed."