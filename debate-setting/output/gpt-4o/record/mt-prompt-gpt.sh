#!/bin/bash
# Script to create and configure a Conda environment for mt-prompt-openai.py

# Define environment name
ENV_NAME="mt_prompt_env"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not in PATH. Please install Conda first."
    echo "Visit https://docs.conda.io/en/latest/miniconda.html for installation instructions."
    exit 1
fi

# Create conda environment with Python 3.9
echo "Creating Conda environment '$ENV_NAME' with Python 3.9..."
conda create -y -n $ENV_NAME python=3.9

# Activate the environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install required packages
echo "Installing required packages..."
conda install -y pandas
pip install litellm  # Using pip since litellm may not be in conda channels

# Create necessary directories
echo "Creating data directory..."
mkdir -p data

# Create sample questions.txt and arguments.txt if they don't exist
if [ ! -f "data/questions.txt" ]; then
    echo "Creating sample questions.txt..."
    cat > data/questions.txt << EOF
Should military funding be increased or decreased?
Is universal healthcare a right or a privilege?
Are social media platforms responsible for the content users post?
Should there be stricter regulations on artificial intelligence development?
EOF
fi

if [ ! -f "data/arguments.txt" ]; then
    echo "Creating sample arguments.txt..."
    cat > data/arguments.txt << EOF
Military funding should be decreased to allocate more resources to social services.
Healthcare is a fundamental human right that should be accessible to all citizens.
Social media platforms should be held accountable for the content on their platforms.
AI development needs strict regulations to prevent potential misuse and ensure ethical implementation.
EOF
fi

# Print instructions
echo ""
echo "===== SETUP COMPLETE ====="
echo "Environment '$ENV_NAME' has been created with the necessary dependencies."
echo ""
echo "To use this environment:"
echo "1. Activate it with: conda activate $ENV_NAME"
echo "2. Run the script with: python mt-prompt-openai.py --api_key YOUR_OPENAI_API_KEY"
echo ""
echo "To deactivate the environment when finished: conda deactivate"
echo ""
echo "Make sure the data/questions.txt and data/arguments.txt files contain your desired questions and arguments."
echo "============================"