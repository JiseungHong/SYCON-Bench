#!/bin/bash

# Script to run mt-prompt.py with specified models
# Created: $(date)

# Set up error handling
set -e

echo "Starting debate-LM model execution script..."

# Change to the repository directory
cd /workspace/debate-LM

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create output directory if it doesn't exist
mkdir -p output

# Run the first model: Qwen/Qwen2.5-72B-Instruct
echo "Running mt-prompt.py with Qwen/Qwen2.5-72B-Instruct..."
python mt-prompt.py "Qwen/Qwen2.5-72B-Instruct"

# Run the second model: meta-llama/Llama-3.3-70B-Instruct
echo "Running mt-prompt.py with meta-llama/Llama-3.3-70B-Instruct..."
python mt-prompt.py "meta-llama/Llama-3.3-70B-Instruct"

echo "All models have been processed successfully!"
echo "Results are available in the output directory."