# Critical Token Analysis

This folder contains tools for identifying and analyzing critical tokens in language model responses during debates. The critical token method helps identify linguistic patterns that differentiate high-quality from low-quality debate responses.

## Method Overview

The critical token analysis uses a contrastive learning approach to identify tokens that are characteristic of effective vs. ineffective debate responses:

1. **Contrastive Fine-tuning**: We fine-tune two separate LoRA adapters on the same base model:
   - A "positive" adapter trained on high-quality debate responses
   - A "negative" adapter trained on low-quality debate responses

2. **Token Scoring**: We compute a contrastive score for each token in negative responses using:
   ```
   score(token) = (1 + β) * log P_pos(token) - β * log P_neg(token)
   ```
   where β is a balancing hyperparameter (default: 1.0)

3. **Analysis**: We identify the most critical tokens (those with the lowest scores) and analyze their distribution across debate turns to understand how language patterns differ between prompting strategies.

## Justification

This method provides several advantages for analyzing debate language:

- **Interpretability**: Identifies specific linguistic features that differentiate good from poor debate responses
- **Efficiency**: Uses parameter-efficient fine-tuning (LoRA) to create specialized models without full retraining
- **Quantitative**: Provides numerical scores that can be used to compare different prompting strategies
- **Generalizable**: Can be applied to any debate format or language model

## Usage

### Step 1: Prepare Data

Place your positive and negative examples in the data folder:
- `analysis/data/positive_trajectory.txt`: High-quality debate responses
- `analysis/data/negative_trajectory.txt`: Low-quality debate responses

Each file should contain one response per line.

### Step 2: Fine-tune LoRA Adapters

```bash
python analysis/finetune.py
```

This will create two LoRA adapters:
- `analysis/model/pos_lora/`: Adapter trained on positive examples
- `analysis/model/neg_lora/`: Adapter trained on negative examples

### Step 3: Generate Contrastive Token Scores

```bash
python analysis/contrastive_token_scores.py
```

This will analyze the negative responses and output a file `contrastive_scores.txt` containing token-level scores.

### Step 4: Analyze Critical Token Usage

```bash
python analysis/analyze_critical_token_usage.py
```

This compares the usage of critical tokens between two different prompting strategies and outputs a CSV file `critical_token_stats.csv` with the results.

## Requirements

Install the required dependencies:

```bash
pip install -r analysis/requirements.txt
```

## References

This method is inspired by research on contrastive learning and token-level analysis in language models. The approach helps identify specific linguistic features that differentiate effective from ineffective debate strategies, providing actionable insights for improving language model prompting.