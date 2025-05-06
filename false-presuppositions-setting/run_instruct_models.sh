#!/bin/bash

# Script to run mt-prompt-pipeline.py with multiple models

# List of models to process
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "google/gemma-2-9b-it"
)

# Parameters
BATCH_SIZE=4
NUM_RESPONSES=5

# Create log directory
mkdir -p logs

# Process each model
for MODEL in "${MODELS[@]}"; do
  # Convert model path to a valid filename for the log
  MODEL_NAME=$(echo $MODEL | sed 's/\//_/g')
  LOG_FILE="logs/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
  
  echo "========================================"
  echo "Starting to process model: $MODEL"
  echo "Log file: $LOG_FILE"
  echo "========================================"
  
  # Run the Python script with the current model and log output
  python mt-prompt-pipeline.py "$MODEL" --batch_size $BATCH_SIZE --num_responses $NUM_RESPONSES 2>&1 | tee "$LOG_FILE"
  
  # Check if the script succeeded
  if [ $? -eq 0 ]; then
    echo "Successfully processed model: $MODEL"
  else
    echo "Error processing model: $MODEL. Check the log file: $LOG_FILE"
  fi
  
  echo "Completed model: $MODEL"
  echo ""
  
  # Optional: Add a short delay between models to allow system to stabilize
  sleep 5
done

echo "All models have been processed."