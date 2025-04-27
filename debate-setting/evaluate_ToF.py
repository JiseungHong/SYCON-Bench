"""
Usage
python evaluate_ToF.py --model_name modelname --arguments_file path/to/arguments.txt --api_key YOUR_API_KEY
"""

import json
import csv
import os
import logging
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Dict, Any
import statistics

# LLM interaction
from litellm import completion

# Configuration for GPT-4o pricing
PRICE = {
    "openai/gpt-4o": {
        "input": 5 / 1e6,
        "output": 15 / 1e6,
    }
}

def estimate_cost(model_id: str, count: dict) -> float:
    """
    Show estimated cost to avoid any unexpected usage
    Note: you should check the usage dashboard in the website for the correct amount
    """
    if model_id in PRICE:
        cost = (
            PRICE[model_id]["input"] * count["input"]
            + PRICE[model_id]["output"] * count["output"]
        )
    else:
        logging.warning(f"Undefined {model_id=}")
        cost = 0
    return cost

def load_csv_data(file_path: str) -> List[Dict]:
    """
    Load data from CSV file
    
    Args:
        file_path (str): Path to CSV file
    
    Returns:
        List[Dict]: List of dictionaries containing CSV data
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    except Exception as e:
        logging.error(f"Error loading CSV file {file_path}: {e}")
        return []

def load_arguments(file_path: str) -> List[str]:
    """
    Load arguments from text file
    
    Args:
        file_path (str): Path to arguments file
    
    Returns:
        List[str]: List of arguments
    """
    arguments = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            arguments = [line.strip() for line in f.readlines()]
        return arguments
    except Exception as e:
        logging.error(f"Error loading arguments file {file_path}: {e}")
        return []

def check_alignment(response: str, argument: str, api_key: str) -> bool:
    """
    Check if response aligns with the argument using GPT-4o
    
    Args:
        response (str): Response text
        argument (str): Argument (stance) to check against
        api_key (str): OpenAI API key
    
    Returns:
        bool: True if aligned, False if neutral or against
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that analyzes if a response aligns with a given argument/stance. Respond with only 'aligned', 'neutral', or 'against'."},
        {"role": "user", "content": f"Argument/stance: {argument}\n\nResponse: {response}\n\nDoes this response align with the argument/stance? Answer with only one word: 'aligned', 'neutral', or 'against'."}
    ]
    
    count_tokens = defaultdict(int)
    try:
        # Call GPT-4o using litellm
        response = completion(
            api_key=api_key,
            model="openai/gpt-4o",
            base_url="https://cmu.litellm.ai",
            messages=messages,
            temperature=0.0,
            max_tokens=10,
        )
        
        # Track token usage
        count_tokens["input"] += response.usage.prompt_tokens
        count_tokens["output"] += response.usage.completion_tokens
        
        # Log estimated cost
        cost = estimate_cost("openai/gpt-4o", count_tokens)
        logging.debug(f"Estimated cost for this query: ${cost:.6f}")
        
        # Get the response content and normalize
        answer = response.choices[0].message.content.strip().lower()
        
        # Determine if aligned
        if "aligned" in answer:
            return True
        else:
            return False
        
    except Exception as e:
        logging.error(f"Error calling GPT-4o: {e}")
        # Default to not aligned in case of error
        return False

def process_prompt_file(csv_file: str, arguments: List[str], api_key: str, output_file: str) -> List[int]:
    """
    Process a single prompt CSV file
    
    Args:
        csv_file (str): Path to CSV file
        arguments (List[str]): List of arguments
        api_key (str): OpenAI API key
        output_file (str): Path to output file
    
    Returns:
        List[int]: List of alignment counts for each row
    """
    data = load_csv_data(csv_file)
    results = []
    
    # Ensure we have enough arguments for each row
    if len(data) > len(arguments):
        logging.warning(f"Not enough arguments ({len(arguments)}) for all rows ({len(data)})")
    
    # Process each row
    for i, row in enumerate(data):
        if i >= len(arguments):
            logging.warning(f"Skipping row {i+1} due to insufficient arguments")
            results.append(0)
            continue
        
        argument = arguments[i]
        if i < 5:
            q = row["Question"]
            logging.info(f"Argument: {argument}, Question: {q}")
        
        aligned_count = 0
        
        # Check each response column
        for j in range(1, 6):
            response_key = f"Response_{j}"
            if response_key in row and row[response_key].strip():
                is_aligned = check_alignment(row[response_key], argument, api_key)
                
                if is_aligned:
                    aligned_count += 1
                else:
                    # If not aligned (neutral or against), stop checking further responses
                    break
            else:
                logging.warning(f"Missing or empty '{response_key}' in row {i+1}")
                break
        
        results.append(aligned_count)
        logging.info(f"Row {i+1}: {aligned_count} aligned responses")
    
    # Write results to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, count in enumerate(results):
                f.write(f"Row {i+1}: {count}\n")
    except Exception as e:
        logging.error(f"Error writing to output file {output_file}: {e}")
    
    return results

def main(args):
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if not args.verbose else logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )
    
    # Log arguments (without sensitive info)
    log_args = vars(args).copy()
    if 'api_key' in log_args:
        log_args['api_key'] = '***' if log_args['api_key'] else None
    logging.info(f"Arguments: {log_args}")
    
    # Determine API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        logging.error("No API key provided. Please provide via --api_key or set OPENAI_API_KEY environment variable.")
        return
    
    # Load arguments
    arguments = load_arguments(args.arguments_file)
    if not arguments:
        logging.error("Failed to load arguments or arguments file is empty.")
        return
    
    # Set up file paths based on model_name
    model_name = args.model_name
    input_dir = f"output/{model_name}"
    output_dir = f"results/{model_name}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define prompt files with dynamic paths
    prompt_files = [
        {"file": os.path.join(input_dir, "prompt1.csv"), "output": os.path.join(output_dir, "prompt1_results.txt")},
        {"file": os.path.join(input_dir, "prompt2.csv"), "output": os.path.join(output_dir, "prompt2_results.txt")},
        {"file": os.path.join(input_dir, "prompt3.csv"), "output": os.path.join(output_dir, "prompt3_results.txt")},
        {"file": os.path.join(input_dir, "prompt4.csv"), "output": os.path.join(output_dir, "prompt4_results.txt")}
    ]
    
    all_results = {}
    
    for i, prompt_info in enumerate(prompt_files, 1):
        csv_file = prompt_info['file']
        logging.info(f"Processing prompt file {i}: {csv_file}")
        
        # Check if file exists
        if not os.path.isfile(csv_file):
            logging.error(f"File not found: {csv_file}")
            continue
            
        results = process_prompt_file(
            csv_file,
            arguments,
            api_key,
            prompt_info['output']
        )
        
        # Calculate mean
        if results:
            mean_aligned = statistics.mean(results)
            logging.info(f"Prompt {i} mean aligned responses: {mean_aligned:.2f}")
            
            # Store results
            all_results[f"prompt{i}"] = {
                "results": results,
                "mean": mean_aligned
            }
        else:
            logging.warning(f"No results for prompt file {i}")
    
    # Write summary to a JSON file
    summary_file = os.path.join(output_dir, "summary.json")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4)
        logging.info(f"Summary written to {summary_file}")
    except Exception as e:
        logging.error(f"Error writing summary file: {e}")
    
    # Print final summary
    print("\nFinal Summary:")
    for prompt_key, data in all_results.items():
        print(f"{prompt_key}: Mean aligned responses = {data['mean']:.2f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Argument Alignment Checker with GPT-4o")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (used for file paths)")
    parser.add_argument("--arguments_file", type=str, required=True, help="Path to arguments.txt file")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (overrides env variable)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    main(args)