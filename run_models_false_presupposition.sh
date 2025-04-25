#!/bin/bash
# Script to run mt-prompt-vllm.py with Qwen2.5-72B-Instruct
# Created: $(date)

# Set up error handling
set -e
echo "Starting ethical multi-turn conversation script..."

cd false-presuppositions-setting

# Create output directory if it doesn't exist
echo "Creating output directory..."
mkdir -p output

# Install requirements (always run this step)
echo "Installing requirements..."
pip install -r requirements-vllm.txt

# Run the model
echo "Running mt-prompt.py with Qwen2.5-72B-Instruct..."
python mt-prompt-vllm.py "Qwen2.5-72B-Instruct" > output.log 2>&1

echo "Model has been processed successfully!"