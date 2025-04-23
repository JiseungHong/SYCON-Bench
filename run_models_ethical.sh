#!/bin/bash
# Script to run mt-prompt.py with Llama-3.1-8B-Instruct
# Created: $(date)

# Set up error handling
set -e
echo "Starting ethical multi-turn conversation script..."

cd ethical-setting

# Create output directory if it doesn't exist
echo "Creating output directory..."
mkdir -p output

# Install requirements (always run this step)
echo "Installing requirements..."
pip install -r requirements.txt

# Run the model
echo "Running mt-prompt.py with meta-llama/Llama-3.1-8B-Instruct..."
python mt-prompt.py "meta-llama/Llama-3.1-8B-Instruct" --output_dir "output/llama-3.1-8b" > output_Llama-3.1-8B-Instruct.log 2>&1

echo "Model has been processed successfully!"
echo "Results are available in the output/llama-3.1-8b directory."