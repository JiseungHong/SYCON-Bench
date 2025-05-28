# SYCON-Bench: Ethical Setting

This directory contains the code for the ethical setting of SYCON-Bench, which evaluates how models respond to questions containing harmful stereotypes when users increasingly pressure them to conform.

## Structure

- `models.py`: Contains the model classes for different types of models (open-source and closed-source)
- `run_benchmark.py`: Main script for running the benchmark
- `data/`: Contains the ethical questions derived from StereoSet
- `output/`: Directory where results are saved

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running the Benchmark

```bash
# For open-source models
python run_benchmark.py "google/gemma-3-12b-it"

# For closed-source models (API-based)
python run_benchmark.py "openai/gpt-4o" --api_key YOUR_API_KEY
```

### Command Line Arguments

- `model_name`: Name or identifier of the model to evaluate
- `--api_key`: API key for closed-source models
- `--base_url`: Custom base URL for API (optional)
- `--batch_size`: Number of questions to process in each batch (default: 4)
- `--output_dir`: Custom output directory (default: "output/{model_id}")
- `--verbose`: Enable verbose logging

## Data Format

The ethical setting uses data from StereoSet, with questions structured as follows:

- Initial question that presents a potentially harmful stereotype
- Follow-up questions that increasingly pressure the model to conform to the stereotype
- Up to 5 turns of conversation per topic

## Metrics

The benchmark measures:

1. **Turn of Flip (ToF)**: How quickly a model adopts a harmful stereotype
2. **Number of Flips (NoF)**: How frequently a model shifts its stance on harmful stereotypes under sustained user pressure