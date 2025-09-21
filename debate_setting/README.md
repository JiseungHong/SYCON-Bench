# SYCON-Bench: Debate Setting

This directory contains the code for the debate setting of SYCON-Bench, which evaluates how models maintain or change their stance on controversial topics when faced with user disagreement.

## Structure

- `models.py`: Contains the model classes for different types of models (open-source and closed-source)
- `run_benchmark.py`: Main script for running the benchmark
- `data/`: Contains the questions and arguments for the debate setting
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
- `--num_responses`: Number of responses to generate for each question (default: 5)
- `--output_dir`: Custom output directory (default: "output/{model_id}")
- `--prompt_type`: Specific prompt type to use (default: "all")
- `--verbose`: Enable verbose logging

### Prompt Types

- `individual_thinker`: Individual Thinker Prompt
- `spt`: Self-Perception Technique Prompt
- `non_sycophantic`: Non-Sycophantic Prompt
- `spt_non_sycophantic`: SPT + Non-Sycophantic Prompt
- `base`: Base prompt (for closed models)

## Metrics

The benchmark measures:

1. **Turn of Flip (ToF)**: How quickly a model conforms to the user's view
2. **Number of Flips (NoF)**: How frequently a model shifts its stance under sustained user pressure
## Model Compatibility Matrix

| Model Family | Supported Versions | Quantization | Chat Template | Dependencies | Known Issues |
|--------------|-------------------|--------------|--------------|--------------|-------------|
| Llama        | All (Llama-2, 3)  | 4-bit        | llama-2      | transformers, bitsandbytes | -           |
| Qwen         | All               | 8-bit        | chatml       | transformers, accelerate   | -           |
| Gemma        | All               | float16      | gemma        | transformers              | -           |
| GPT (OpenAI) | All               | API          | openai       | litellm                  | -           |
| Claude       | All               | API          | anthropic    | litellm                  | -           |

Add new models to this table and to `model_registry.py` for full support.

## Model Testing

Run `pytest tests/` to verify model registry and compatibility logic. Add new tests for each new model family.

