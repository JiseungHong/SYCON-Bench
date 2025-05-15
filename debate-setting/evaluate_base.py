"""
Usage
python evaluate_base.py --model_name modelname --api_key YOUR_API_KEY
"""

import json
import csv
import os
import logging
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Dict, Any

# LLM interaction
from openai import AzureOpenAI

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

def check_alignment(client, response: str, argument: str) -> bool:
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
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=10,
        )
        
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

def process_prompt_file(csv_file: str, arguments: List[str], api_key: str, output_file: str) -> Dict[str, Any]:
    """
    Process the prompt.csv file that only has Response_1 and Response_2
    
    Args:
        csv_file (str): Path to CSV file
        arguments (List[str]): List of arguments
        api_key (str): OpenAI API key
        output_file (str): Path to output file
    
    Returns:
        Dict[str, Any]: Dictionary with results for both Response_1 and Response_2
    """
    data = load_csv_data(csv_file)
    response1_results = []
    response2_results = []
    response2_conditional_results = []
    
    # Process each row
    for i, row in enumerate(data):
        if i >= len(arguments):
            logging.warning(f"Skipping row {i+1} due to insufficient arguments")
            response1_results.append(0)
            response2_results.append(0)
            continue
        
        argument = arguments[i]
        question = row.get("Question", f"Row {i+1}")
        if i < 5:
            logging.info(f"Argument: {argument}, Question: {question}")
        
        # Use AzureOpenAI for API calls
        client = AzureOpenAI(
            api_key = api_key,
            api_version = "2023-05-15",
            azure_endpoint = "https://gpt-35-1106.openai.azure.com/"
        )
        
        # Check Response_1
        if "Response_1" in row and row["Response_1"].strip():
            is_aligned1 = check_alignment(client, row["Response_1"], argument)
            response1_results.append(1 if is_aligned1 else 0)
            logging.info(f"Row {i+1} Response_1: {'Aligned' if is_aligned1 else 'Not aligned'}")
            
            # Check Response_2 only if Response_1 is available
            if "Response_2" in row and row["Response_2"].strip():
                is_aligned2 = check_alignment(client, row["Response_2"], argument)
                response2_results.append(1 if is_aligned2 else 0)
                
                # Track Response_2 alignment only when Response_1 was aligned
                if is_aligned1:
                    response2_conditional_results.append(1 if is_aligned2 else 0)
                
                logging.info(f"Row {i+1} Response_2: {'Aligned' if is_aligned2 else 'Not aligned'}")
            else:
                response2_results.append(0)
                logging.warning(f"Missing or empty 'Response_2' in row {i+1}")
        else:
            logging.warning(f"Missing or empty 'Response_1' in row {i+1}")
            response1_results.append(0)
            response2_results.append(0)
    
    # Calculate stats
    total_questions = len(response1_results)
    response1_aligned = sum(response1_results)
    response2_aligned = sum(response2_results)
    
    # Calculate conditional alignment (Response_2 aligned when Response_1 was aligned)
    response2_conditional_count = len(response2_conditional_results)
    response2_conditional_aligned = sum(response2_conditional_results) if response2_conditional_results else 0
    
    # Create results dictionary
    results = {
        "total_questions": total_questions,
        "response1": {
            "aligned_count": response1_aligned,
            "aligned_percentage": (response1_aligned / total_questions * 100) if total_questions > 0 else 0
        },
        "response2": {
            "aligned_count": response2_aligned,
            "aligned_percentage": (response2_aligned / total_questions * 100) if total_questions > 0 else 0
        },
        "conditional": {
            "count": response2_conditional_count,
            "aligned_count": response2_conditional_aligned,
            "aligned_percentage": (response2_conditional_aligned / response2_conditional_count * 100) 
                                 if response2_conditional_count > 0 else 0
        },
        "raw_results": {
            "response1": response1_results,
            "response2": response2_results
        }
    }
    
    # Write results to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results written to {output_file}")
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
    arguments = load_arguments("data/arguments.txt")
    if not arguments:
        logging.error("Failed to load arguments or arguments file is empty.")
        return
    
    # Set up file paths based on model_name
    model_name = args.model_name
    input_dir = f"output/{model_name}"
    output_dir = f"results/{model_name}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define prompt file path
    csv_file = os.path.join(input_dir, "prompt.csv")
    output_file = os.path.join(output_dir, "prompt_results.json")
    
    # Check if file exists
    if not os.path.isfile(csv_file):
        logging.error(f"File not found: {csv_file}")
        return
        
    # Process prompt file
    logging.info(f"Processing prompt file: {csv_file}")
    results = process_prompt_file(
        csv_file,
        arguments,
        api_key,
        output_file
    )
    
    # Print final summary to CLI
    print("\nEvaluation Results:")
    print("=" * 60)
    print(f"Total questions evaluated: {results['total_questions']}")
    print(f"Response_1 aligned count: {results['response1']['aligned_count']} ({results['response1']['aligned_percentage']:.2f}%)")
    print(f"Response_2 aligned count: {results['response2']['aligned_count']} ({results['response2']['aligned_percentage']:.2f}%)")
    print(f"Response_2 aligned count (when Response_1 was aligned): {results['conditional']['aligned_count']} out of {results['conditional']['count']} ({results['conditional']['aligned_percentage']:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    parser = ArgumentParser(description="Argument Alignment Checker for Base Models")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (used for file paths)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (overrides env variable)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    main(args)