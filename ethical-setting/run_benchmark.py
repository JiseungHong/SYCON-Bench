"""
Main script for running the SYCON-Bench ethical setting benchmark.

This script provides a unified interface for running the benchmark with different models.
"""
import os
import argparse
import logging
import time
import csv
import pandas as pd
from models import ModelFactory
import torch

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)


def read_data(data_file="data/stereoset_intra_user_queries_api_over45.csv"):
    """Read ethical questions from data file."""
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def run_benchmark(model, data, batch_size=4, output_dir=None):
    """
    Run the benchmark for a given model on the provided ethical questions.
    
    Args:
        model: The model instance to use
        data: List of question data dictionaries
        batch_size: Number of questions to process in each batch
        output_dir: Directory to save results
    """
    results = []
    
    # Process questions in batches
    for batch_start in range(0, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        logging.info(f"\nProcessing questions {batch_start+1}-{batch_end} out of {len(data)}")
        
        for item in batch_data:
            result = {
                'target': item['target'],
                'bias_type': item['bias_type'],
                'stereotype': item['stereotype'],
                'question': item['question']
            }
            
            # First question
            messages = model.get_chat_messages(item['question'])
            response1 = model.generate_responses(messages)
            result['response1'] = response1
            
            # Second question (follow-up)
            if item['question2']:
                messages = model.get_chat_messages(item['question'], item['question2'])
                response2 = model.generate_responses(messages)
                result['response2'] = response2
            
            # Third question (follow-up)
            if item['question3']:
                messages = model.get_chat_messages(item['question'], item['question2'], item['question3'])
                response3 = model.generate_responses(messages)
                result['response3'] = response3
            
            # Fourth question (follow-up)
            if item['question4']:
                messages = model.get_chat_messages(item['question'], item['question2'], item['question3'], item['question4'])
                response4 = model.generate_responses(messages)
                result['response4'] = response4
            
            # Fifth question (follow-up)
            if item['question5']:
                messages = model.get_chat_messages(item['question'], item['question2'], item['question3'], item['question4'], item['question5'])
                response5 = model.generate_responses(messages)
                result['response5'] = response5
            
            results.append(result)
        
        # Save batch results
        output_file = f"{output_dir}/results.csv"
        model.save_results(results, output_file)
        logging.info(f"Saved results to {output_file}")
        
        logging.info(f"Completed batch {batch_start//batch_size + 1} of {(len(data)-1)//batch_size + 1}")
    
    # Log completion
    logging.info(f"\nCompleted processing model")
    logging.info(f"Final results saved to {output_dir}/results.csv")

def main():
    parser = argparse.ArgumentParser(description="Run SYCON-Bench ethical setting benchmark")
    parser.add_argument("model_name", type=str, help="Model name or identifier")
    parser.add_argument("--api_key", type=str, help="API key for closed models")
    parser.add_argument("--base_url", type=str, help="Base URL for API (for closed models)")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of questions to process in each batch")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
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
    
    # Create a shorter model identifier for directory names
    model_id = args.model_name.split("/")[-1]
    
    # Ensure the output directory exists
    output_dir = args.output_dir if args.output_dir else f"output/{model_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Log information about the run
    logging.info(f"Processing model: {args.model_name}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Batch size: {args.batch_size}")
    
    # Create the model
    model = ModelFactory.create_model(args.model_name, api_key=args.api_key, base_url=args.base_url,device=device)
    
    # Set up the model
    model.setup()
    
    # Read the data
    data = read_data()
    
    # Run the benchmark
    run_benchmark(
        model=model,
        data=data,
        batch_size=args.batch_size,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()