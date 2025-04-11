#!/bin/bash

# Advanced script to run mt-prompt.py with specified models
# Created: $(date)

# Set up error handling
set -e

# Default values
BATCH_SIZE=4
NUM_RESPONSES=5
OUTPUT_DIR="output"
INSTALL_DEPS=true

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run mt-prompt.py with specified models"
    echo ""
    echo "Options:"
    echo "  -b, --batch-size SIZE     Set batch size (default: 4)"
    echo "  -r, --responses NUM       Number of responses per question (default: 5)"
    echo "  -o, --output-dir DIR      Custom output directory (default: output)"
    echo "  -s, --skip-install        Skip installing dependencies"
    echo "  -h, --help                Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -r|--responses)
            NUM_RESPONSES="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--skip-install)
            INSTALL_DEPS=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

echo "Starting debate-LM model execution script..."

# Change to the repository directory
cd /workspace/debate-LM

# Install requirements if not skipped
if [ "$INSTALL_DEPS" = true ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Skipping dependency installation..."
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define models to run
MODELS=(
    "Qwen/Qwen2.5-72B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
)

# Run each model
for MODEL in "${MODELS[@]}"; do
    echo "Running mt-prompt.py with $MODEL..."
    
    # Construct command with options
    CMD="python mt-prompt.py \"$MODEL\" --batch_size $BATCH_SIZE --num_responses $NUM_RESPONSES"
    
    # Add output directory if custom
    if [ "$OUTPUT_DIR" != "output" ]; then
        CMD="$CMD --output_dir $OUTPUT_DIR"
    fi
    
    echo "Executing: $CMD"
    eval $CMD
    
    echo "Completed processing $MODEL"
    echo "----------------------------------------"
done

echo "All models have been processed successfully!"
echo "Results are available in the $OUTPUT_DIR directory."