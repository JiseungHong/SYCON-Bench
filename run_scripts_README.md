# Model Execution Scripts

This directory contains scripts to run the `mt-prompt.py` with specified models.

## Basic Script: `run_models.sh`

This script runs `mt-prompt.py` with the following models:
- Qwen/Qwen2.5-72B-Instruct
- meta-llama/Llama-3.3-70B-Instruct

### Usage

```bash
./run_models.sh
```

The script will:
1. Install all dependencies from `requirements.txt`
2. Run each model sequentially with default parameters
3. Save results to the `output` directory

## Advanced Script: `run_models_advanced.sh`

This script provides more flexibility with command-line options.

### Usage

```bash
./run_models_advanced.sh [OPTIONS]
```

### Options

- `-b, --batch-size SIZE`: Set batch size (default: 4)
- `-r, --responses NUM`: Number of responses per question (default: 5)
- `-o, --output-dir DIR`: Custom output directory (default: output)
- `-s, --skip-install`: Skip installing dependencies
- `-h, --help`: Display help message

### Examples

Run with default settings:
```bash
./run_models_advanced.sh
```

Run with custom batch size and number of responses:
```bash
./run_models_advanced.sh --batch-size 2 --responses 3
```

Run with custom output directory and skip dependency installation:
```bash
./run_models_advanced.sh --output-dir custom_results --skip-install
```

## Notes

- Both scripts will create the output directory if it doesn't exist
- The scripts run the models sequentially, which may take a significant amount of time
- Results are saved in CSV format in the output directory