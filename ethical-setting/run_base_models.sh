#!/bin/bash

# Models to process
MODELS=(
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-14B"
  "Qwen/Qwen2.5-72B"
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-70B"
  "google/gemma-2-9b"
)

# Set batch size
BATCH_SIZE=4

# Set number of responses
NUM_RESPONSES=5

# Log file for the script
LOG_FILE="run_models_$(date +%Y%m%d_%H%M%S).log"

echo "Starting model processing at $(date)" | tee -a $LOG_FILE

# Process each model
for model in "${MODELS[@]}"; do
  echo "=======================================" | tee -a $LOG_FILE
  echo "Processing model: $model" | tee -a $LOG_FILE
  echo "Started at: $(date)" | tee -a $LOG_FILE
  
  # Create model-specific log filename
  model_name=$(echo $model | tr '/' '_')
  
  # Run the Python script
  echo "Running: python mt-prompt-base-models.py \"$model\" --batch_size $BATCH_SIZE --num_responses $NUM_RESPONSES" | tee -a $LOG_FILE
  
  python mt-prompt-base-models.py "$model" --batch_size $BATCH_SIZE --num_responses $NUM_RESPONSES 2>&1 | tee -a "model_${model_name}.log"
  
  # Check if the process was successful
  if [ $? -eq 0 ]; then
    echo "Successfully processed model: $model" | tee -a $LOG_FILE
  else
    echo "ERROR: Failed to process model: $model" | tee -a $LOG_FILE
  fi
  
  echo "Finished at: $(date)" | tee -a $LOG_FILE
  echo "=======================================" | tee -a $LOG_FILE
  echo "" | tee -a $LOG_FILE
  
  # Sleep for a minute between models to avoid potential resource issues
  echo "Waiting 60 seconds before processing next model..." | tee -a $LOG_FILE
  sleep 60
done

echo "All models processing completed at $(date)" | tee -a $LOG_FILE