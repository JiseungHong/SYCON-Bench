#!/bin/bash

# Define the list of models to evaluate
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "google/gemma-2-9b-it"
)

# Set default parameters
PRESUPPOSITIONS_FILE="data/presuppositions.txt"
BATCH_SIZE=10

# Create a log directory
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Run the evaluation for each model
for MODEL in "${MODELS[@]}"; do
  echo "Processing model: $MODEL"
  
  # Extract the model ID for log naming
  MODEL_ID=$(echo $MODEL | sed 's/.*\///')
  LOG_FILE="$LOG_DIR/${MODEL_ID}_evaluation.log"
  
  # Run the evaluation script and log the output
  echo "Starting evaluation at $(date)" > $LOG_FILE
  python evaluate_presuppositions.py "$MODEL" \
    --presuppositions_file $PRESUPPOSITIONS_FILE \
    --batch_size $BATCH_SIZE 2>&1 | tee -a $LOG_FILE
  
  echo "Completed evaluation for $MODEL at $(date)" >> $LOG_FILE
  echo "Results saved to results/$MODEL_ID/"
  echo "------------------------------------------"
done

# Summarize all model results
echo "Generating combined summary..."
echo "Model Evaluation Summary" > combined_results.txt
echo "=========================" >> combined_results.txt
echo "" >> combined_results.txt

for MODEL in "${MODELS[@]}"; do
  MODEL_ID=$(echo $MODEL | sed 's/.*\///')
  FACTUAL_CHECK_FILE="results/$MODEL_ID/factual_check.txt"
  
  if [ -f "$FACTUAL_CHECK_FILE" ]; then
    echo "Results for $MODEL_ID:" >> combined_results.txt
    grep "Accuracy" "$FACTUAL_CHECK_FILE" >> combined_results.txt
    echo "" >> combined_results.txt
  else
    echo "No results found for $MODEL_ID" >> combined_results.txt
    echo "" >> combined_results.txt
  fi
done

echo "All evaluations completed!"
echo "Individual model results are in the results directory"
echo "Combined summary available in combined_results.txt"
echo "Logs are saved in the logs directory"