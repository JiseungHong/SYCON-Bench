#!/bin/bash

# Script to run mt-prompt-pipeline.py with multiple models and different seed values

# List of models to process
MODELS=(
  "Qwen/Qwen2.5-14B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

# Parameters
BATCH_SIZE=4
NUM_RESPONSES=5

# Seed values to use
SEEDS=(307 411)

# Create log directory
mkdir -p logs

# Process each model with each seed
for MODEL in "${MODELS[@]}"; do
  # Convert model path to a valid filename for the log
  MODEL_NAME=$(echo $MODEL | sed 's/\//_/g')
  
  for SEED in "${SEEDS[@]}"; do
    LOG_FILE="logs/${MODEL_NAME}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "========================================"
    echo "Starting to process model: $MODEL with seed: $SEED"
    echo "Log file: $LOG_FILE"
    echo "========================================"
    
    # Run the Python script with the current model, seed, and log output
    # Note: Added --seed parameter to pass the seed value to the Python script
    python mt-prompt-pipeline.py "$MODEL" --batch_size $BATCH_SIZE --num_responses $NUM_RESPONSES --seed $SEED 2>&1 | tee "$LOG_FILE"
    
    # Check if the script succeeded
    if [ $? -eq 0 ]; then
      echo "Successfully processed model: $MODEL with seed: $SEED"
    else
      echo "Error processing model: $MODEL with seed: $SEED. Check the log file: $LOG_FILE"
    fi
    
    echo "Completed model: $MODEL with seed: $SEED"
    echo ""
    
    # Optional: Add a short delay between runs to allow system to stabilize
    sleep 5
  done
done

echo "All models have been processed with all seed values."