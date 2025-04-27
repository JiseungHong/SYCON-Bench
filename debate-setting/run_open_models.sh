#!/bin/bash

# Install requirements first
pip install -r requirements.txt

# Array of models to process, ordered from smaller to larger
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-9b-it"
    "Qwen/Qwen2.5-14B-Instruct"
    "google/gemma-3-12b-it"
    "meta-llama/Llama-3.3-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "google/gemma-3-27b-it"
)

# Loop through models and run the script
for MODEL in "${MODELS[@]}"; do
    echo "Starting processing for model: $MODEL"
    
    # Run the Python script for the current model
    python mt-prompt-open-models.py "$MODEL"
    
    # Check if the script was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed $MODEL"
    else
        echo "Failed to process $MODEL"
        # Optionally, you can choose to break or continue
        # break
    fi
    
    # Optional: add a small delay between model runs
    sleep 30
done

echo "Completed processing all models."