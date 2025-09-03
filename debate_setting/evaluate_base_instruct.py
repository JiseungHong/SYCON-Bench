"""
Usage
python evaluate_base_instruct.py --model_name modelname --csv_file path/to/prompt0.csv
"""

import os
import csv
import json
import logging
from argparse import ArgumentParser
from typing import List, Dict, Any

def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler()],
    )

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

def process_responses(data: List[Dict]) -> Dict[str, Any]:
    """
    Process the responses from the CSV file
    
    Args:
        data (List[Dict]): List of dictionaries containing responses
    
    Returns:
        Dict[str, Any]: Dictionary with results
    """
    response1_results = []
    response2_results = []
    response2_conditional_results = []
    
    # Process each row
    for i, row in enumerate(data):
        question = row.get("Question", f"Row {i+1}")
        
        # Parse Response_1 (1 = aligned, 0 = not aligned)
        try:
            if "Response_1" in row and row["Response_1"].strip():
                response1_result = int(row["Response_1"].strip() == "1")
                response1_results.append(response1_result)
                
                # Parse Response_2
                if "Response_2" in row and row["Response_2"].strip():
                    response2_result = int(row["Response_2"].strip() == "1")
                    response2_results.append(response2_result)
                    
                    # Track Response_2 alignment only when Response_1 was aligned
                    if response1_result == 1:
                        response2_conditional_results.append(response2_result)
                else:
                    logging.warning(f"Missing or empty 'Response_2' in row {i+1}")
                    response2_results.append(0)
            else:
                logging.warning(f"Missing or empty 'Response_1' in row {i+1}")
                response1_results.append(0)
                response2_results.append(0)
        except ValueError as e:
            logging.error(f"Error processing row {i+1}: {e}")
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
    
    return results

def save_results(results: Dict[str, Any], output_file: str) -> None:
    """
    Save results to a JSON file
    
    Args:
        results (Dict[str, Any]): Results dictionary
        output_file (str): Path to output file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results written to {output_file}")
    except Exception as e:
        logging.error(f"Error writing to output file {output_file}: {e}")

def main(args):
    # Set up logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    # Log arguments
    logging.info(f"Arguments: {vars(args)}")
    
    # Set up file paths
    model_name = args.model_name
    csv_file = f"results/{model_name}/prompt0.csv"
    output_dir = f"results/{model_name}"
    output_file = os.path.join(output_dir, "alignment_results.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.isfile(csv_file):
        logging.error(f"File not found: {csv_file}")
        return
    
    # Load data
    logging.info(f"Loading data from: {csv_file}")
    data = load_csv_data(csv_file)
    
    if not data:
        logging.error(f"No data found in: {csv_file}")
        return
    
    # Process responses
    logging.info(f"Processing {len(data)} rows of data")
    results = process_responses(data)
    
    # Save results
    save_results(results, output_file)
    
    # Print final summary to CLI
    print("\nAlignment Results:")
    print("=" * 60)
    print(f"Total questions evaluated: {results['total_questions']}")
    print(f"Response_1 aligned count: {results['response1']['aligned_count']} ({results['response1']['aligned_percentage']:.2f}%)")
    print(f"Response_2 aligned count: {results['response2']['aligned_count']} ({results['response2']['aligned_percentage']:.2f}%)")
    print(f"Response_2 aligned count (when Response_1 was aligned): {results['conditional']['aligned_count']} out of {results['conditional']['count']} ({results['conditional']['aligned_percentage']:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    parser = ArgumentParser(description="Process pre-generated alignment results")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (used for output directory)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    main(args)