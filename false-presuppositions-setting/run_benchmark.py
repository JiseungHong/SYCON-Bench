"""
Main script for running the SYCON-Bench false presuppositions setting benchmark.

This script provides a unified interface for running the benchmark with different models.
"""
import os
import argparse
import logging
import time
import sys
from models import ModelFactory
from data_validation import PresuppositionDataValidator, DataValidationError

def read_data(data_file="data/questions.txt", validate=True):
    """
    Read questions from data file with optional validation.

    Args:
        data_file: Path to the questions file
        validate: Whether to perform comprehensive validation

    Returns:
        List of questions

    Raises:
        DataValidationError: If validation fails
    """
    if validate:
        # Use comprehensive validation
        data_dir = os.path.dirname(data_file) or "data"
        filename = os.path.basename(data_file)
        validator = PresuppositionDataValidator(data_dir)
        try:
            questions = validator.validate_presupposition_data(filename)
            return questions
        except DataValidationError as e:
            logging.error(f"Data validation failed: {e}")
            sys.exit(1)
    else:
        # Legacy behavior with minimal validation
        try:
            with open(data_file, "r", encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]

            if not questions:
                logging.error(f"No questions found in file: {data_file}")
                sys.exit(1)

            return questions
        except FileNotFoundError:
            logging.error(f"Questions file not found: {data_file}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error reading questions file {data_file}: {e}")
            sys.exit(1)

def run_benchmark(model, questions, batch_size=4, output_dir=None, prompt_types=None):
    """
    Run the benchmark for a given model on the provided questions.
    
    Args:
        model: The model instance to use
        questions: List of questions
        batch_size: Number of questions to process in each batch
        output_dir: Directory to save results
        prompt_types: List of prompt types to use (default: all prompt types)
    """
    if prompt_types is None:
        prompt_types = [
            ("base", "base"),
            ("critical", "critical"),
            ("presupposition", "presupposition")
        ]
    
    # Initialize result dictionaries for each prompt type
    results = {prompt_file: {q: [] for q in questions} for _, prompt_file in prompt_types}
    
    # Process questions in batches
    for batch_start in range(0, len(questions), batch_size):
        batch_end = min(batch_start + batch_size, len(questions))
        batch_questions = questions[batch_start:batch_end]
        
        logging.info(f"\nProcessing questions {batch_start+1}-{batch_end} out of {len(questions)}")
        
        # Process each prompt type for the current batch
        for prompt_type, prompt_file in prompt_types:
            logging.info(f"Processing {prompt_type} prompt...")
            for question in batch_questions:
                messages = model.get_chat_messages(question, prompt_type)
                response = model.generate_responses(messages)
                results[prompt_file][question] = [response]  # Wrap in list for consistency with other settings
            
            # Save batch results
            model.save_batch_results(results[prompt_file], f"{output_dir}/{prompt_file}.csv", batch_questions)
            logging.info(f"Saved batch results to {output_dir}/{prompt_file}.csv")
        
        logging.info(f"Completed batch {batch_start//batch_size + 1} of {(len(questions)-1)//batch_size + 1}")
    
    # Log completion
    prompt_files = [prompt_file for _, prompt_file in prompt_types]
    logging.info(f"\nCompleted processing model")
    logging.info(f"Final results saved to {', '.join([f'{output_dir}/{pf}.csv' for pf in prompt_files])}")

def main():
    parser = argparse.ArgumentParser(description="Run SYCON-Bench false presuppositions setting benchmark")
    parser.add_argument("model_name", type=str, help="Model name or identifier")
    parser.add_argument("--api_key", type=str, help="API key for closed models")
    parser.add_argument("--base_url", type=str, help="Base URL for API (for closed models)")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of questions to process in each batch")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--prompt_type", type=str, choices=["all", "base", "critical", "presupposition"], 
                        default="all", help="Prompt type to use")
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
    model = ModelFactory.create_model(args.model_name, api_key=args.api_key, base_url=args.base_url)
    
    # Set up the model
    model.setup()
    
    # Read the data
    questions = read_data()
    
    # Determine which prompt types to use
    if args.prompt_type == "all":
        prompt_types = [
            ("base", "base"),
            ("critical", "critical"),
            ("presupposition", "presupposition")
        ]
    else:
        prompt_types = [(args.prompt_type, args.prompt_type)]
    
    # Run the benchmark
    run_benchmark(
        model=model,
        questions=questions,
        batch_size=args.batch_size,
        output_dir=output_dir,
        prompt_types=prompt_types
    )

if __name__ == "__main__":
    main()
