# SYCON-Bench: Measuring Sycophancy of Language Models in Multi-turn Dialogues

SYCON-Bench is a novel benchmark for evaluating sycophantic behavior in multi-turn, free-form conversational settings. This benchmark measures how quickly a model conforms to the user (*Turn of Flip*) and how frequently it shifts its stance under sustained user pressure (*Number of Flip*).

## Overview

Large Language Models (LLMs) are expected to provide helpful and harmless responses, yet they often exhibit *sycophancy*—conforming to user beliefs regardless of factual accuracy or ethical soundness. Prior research on sycophancy has primarily focused on single-turn factual correctness, overlooking the dynamics of real-world interactions.

SYCON-Bench addresses this gap by:
1. Evaluating models in multi-turn, free-form conversational settings
2. Measuring sycophancy across three real-world scenarios
3. Providing metrics to quantify sycophantic behavior

## Benchmark Structure

SYCON-Bench consists of three distinct settings:

### 1. Debate Setting

Tests how models maintain or change their stance on controversial topics when faced with user disagreement.

- **Data**: 100 controversial debate topics with predefined stances
- **Metrics**: Turn of Flip (ToF), Number of Flips (NoF)
- **Prompting Strategies**: Individual Thinker, Self-Perception Technique (SPT), Non-Sycophantic, SPT+Non-Sycophantic

### 2. Ethical Setting

Evaluates how models respond to questions containing harmful stereotypes when users increasingly pressure them to conform.

- **Data**: Ethical questions derived from StereoSet
- **Metrics**: Turn of Flip (ToF), Number of Flips (NoF)
- **Evaluation**: Measures how quickly models adopt harmful stereotypes

### 3. False Presuppositions Setting

Tests how models handle questions with false presuppositions and whether they maintain factual accuracy under user pressure.

- **Data**: Questions containing false presuppositions
- **Metrics**: Turn of Flip (ToF), Number of Flips (NoF)
- **Evaluation**: Measures how models respond to factually incorrect assumptions

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/username/SYCON-Bench.git
cd SYCON-Bench

# Install dependencies for each setting
cd debate-setting
pip install -r requirements.txt
cd ../ethical-setting
pip install -r requirements.txt
cd ../false-presuppositions-setting
pip install -r requirements.txt
```

### Running the Benchmark

Each setting has a unified interface for running the benchmark:

#### Debate Setting

```bash
cd debate-setting

# For open-source models
python run_benchmark.py "google/gemma-3-12b-it"

# For closed-source models (API-based)
python run_benchmark.py "openai/gpt-4o" --api_key YOUR_API_KEY
```

#### Ethical Setting

```bash
cd ethical-setting

# For open-source models
python run_benchmark.py "google/gemma-3-12b-it"

# For closed-source models (API-based)
python run_benchmark.py "openai/gpt-4o" --api_key YOUR_API_KEY
```

#### False Presuppositions Setting

```bash
cd false-presuppositions-setting

# For open-source models
python run_benchmark.py "google/gemma-3-12b-it"

# For closed-source models (API-based)
python run_benchmark.py "openai/gpt-4o" --api_key YOUR_API_KEY
```

### Command Line Arguments

Each `run_benchmark.py` script supports the following arguments:

- `model_name`: Name or identifier of the model to evaluate
- `--api_key`: API key for closed-source models
- `--base_url`: Custom base URL for API (optional)
- `--batch_size`: Number of questions to process in each batch (default: 4)
- `--output_dir`: Custom output directory (default: "output/{model_id}")
- `--prompt_type`: Specific prompt type to use (default: "all")
- `--verbose`: Enable verbose logging

## Key Findings

Our analysis of 17 LLMs across the three settings revealed:

1. **Alignment tuning amplifies sycophancy**: Models fine-tuned with RLHF or other alignment techniques show increased sycophantic behavior.

2. **Model scaling reduces sycophancy**: Larger models generally demonstrate greater resistance to user pressure.

3. **Reasoning optimization helps**: Models optimized for reasoning abilities show improved resistance to sycophancy.

4. **Third-person perspective reduces sycophancy**: Adopting a third-person perspective (SPT) reduces sycophancy by up to 63.8% in the debate setting.

## Citation

If you use SYCON-Bench in your research, please cite our paper:

```
@article{sycon2025,
  title={Measuring Sycophancy of Language Models in Multi-turn Dialogues},
  author={...},
  journal={...},
  year={2025}
}
```

## License

[MIT License](LICENSE)
