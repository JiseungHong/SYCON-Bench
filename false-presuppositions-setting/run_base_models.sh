#!/bin/bash

# Array of models to process
MODELS=(
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-14B"
  "Qwen/Qwen2.5-72B"
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-70B"
  "google/gemma-2-9b"
)

# Function to display usage
function display_usage {
  echo "Usage: $0 [OPTIONS]"
  echo "OPTIONS:"
  echo "  -m, --model <model_name>   Run for a specific model only"
  echo "  -h, --help                 Display this help message"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      # Run only a specific model
      MODELS=("$2")
      shift 2
      ;;
    -h|--help)
      display_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      display_usage
      exit 1
      ;;
  esac
done

# # Check for required python dependencies
# echo "Checking dependencies..."
# pip install -q transformers torch pandas tqdm

# Create data directory if it doesn't exist
mkdir -p data

# Make sure questions.txt and push_back.csv exist
if [ ! -f "data/questions.txt" ]; then
  echo "Error: data/questions.txt not found!"
  exit 1
fi

if [ ! -f "data/push_back.csv" ]; then
  echo "Error: data/push_back.csv not found!"
  exit 1
fi

# Run the script for each model
for MODEL in "${MODELS[@]}"; do
  echo "====================================================================="
  echo "Processing model: $MODEL"
  echo "====================================================================="
  
  # Extract model id for output directory
  MODEL_ID=$(echo $MODEL | awk -F'/' '{print $NF}')
  
  # Create output directory
  mkdir -p "output/$MODEL_ID"
  
  # Run the script
  echo "Running mt-prompt-base-models.py for $MODEL..."
  python mt-prompt-base-models.py "$MODEL"
  
  # Check if the script executed successfully
  if [ $? -eq 0 ]; then
    echo "Successfully processed $MODEL"
  else
    echo "Error processing $MODEL"
  fi
  
  echo "-------------------------------------------------------------------"
done

echo "All models processed!"