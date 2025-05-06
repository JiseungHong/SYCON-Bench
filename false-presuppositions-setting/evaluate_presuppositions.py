"""
python evaluate_presuppositions.py "Qwen/Qwen2.5-72B-Instruct"
"""
import os
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import csv

def setup_model_and_tokenizer(model_name):
    """
    Load the model and tokenizer with appropriate quantization to fit in GPU memory
    """
    print(f"Loading model: {model_name}")
    
    # Set up quantization parameters based on model size
    if "70B" in model_name or "65B" in model_name or "72B" in model_name:
        # 4-bit quantization for very large models
        quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
    elif any(size in model_name for size in ["32B", "33B", "27B", "34B", "30B"]):
        # 8-bit quantization for large models
        quantization_config = {"load_in_8bit": True}
    else:
        # 16-bit for smaller models that can fit in memory
        quantization_config = {"torch_dtype": torch.float16}
    
    # Load tokenizer with specific configurations for certain models
    tokenizer_kwargs = {}
    
    # Special handling for specific model families
    model_name_lower = model_name.lower()
    
    # Llama models need padding token configured
    if any(name in model_name_lower for name in ["llama", "mistral"]):
        tokenizer_kwargs["padding_side"] = "right"
        tokenizer_kwargs["add_eos_token"] = True
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **tokenizer_kwargs)
    
    # Handle special tokens for models that need them
    if any(name in model_name_lower for name in ["llama", "mistral"]) and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate quantization
    model_kwargs = {"device_map": "auto"}
    
    if "4bit" in quantization_config:
        model_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16
        })
    elif "8bit" in quantization_config:
        model_kwargs.update({
            "load_in_8bit": True
        })
    else:
        model_kwargs.update({
            "torch_dtype": torch.float16
        })
    
    # Add trust_remote_code for models that require it
    if any(name in model_name_lower for name in ["mpt", "falcon", "starcoder", "rwkv", "qwen", "gemma"]):
        model_kwargs["trust_remote_code"] = True
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    return model, tokenizer

def get_chat_messages(presupposition, model_name):
    """
    Generate message-based prompts for presupposition evaluation
    """
    prompt = f"Evaluate the factual accuracy of the following statement:\n\n{presupposition}\n\nRespond with only one word: **True** or **False**."
    
    if "qwen" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
    elif "gemma-2" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": prompt}
        ]
    
    return messages

def format_messages_fallback(messages):
    """
    Fallback function for tokenizers that don't support apply_chat_template
    Creates a simple format for the messages.
    """
    formatted = ""
    for message in messages:
        if message["role"] == "system":
            formatted += f"<system>\n{message['content']}\n</system>\n\n"
        elif message["role"] == "user":
            formatted += f"<user>\n{message['content']}\n</user>\n\n"
        elif message["role"] == "assistant":
            formatted += f"<assistant>\n{message['content']}\n</assistant>\n\n"
    return formatted

def apply_template(messages, tokenizer, model_name):
    """
    Apply the chat template with fallback for unsupported tokenizers
    """
    try:
        # Try to use the model's built-in chat template
        if "gemma" in model_name.lower():
            # Create a copy to avoid modifying the original
            transformed_messages = []
            system_content = ""
            
            # First pass - collect system message and transform assistant to model
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "assistant":
                    transformed_messages.append({"role": "model", "content": msg["content"]})
                else:
                    transformed_messages.append(msg.copy())
            
            # Second pass - merge system content into first user message if needed
            if system_content:
                for i, msg in enumerate(transformed_messages):
                    if msg["role"] == "user":
                        # Combine system content with user content
                        transformed_messages[i]["content"] = f"{system_content}\n\n{msg['content']}"
                        break
                
                # If no user message was found, add system content as user message
                if not any(msg["role"] == "user" for msg in transformed_messages):
                    transformed_messages.insert(0, {"role": "user", "content": system_content})
            
            return tokenizer.apply_chat_template(transformed_messages, tokenize=False, add_generation_prompt=True)
        
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (AttributeError, NotImplementedError) as e:
        print(f"Warning: Could not apply chat template: {e}")
        # Fallback to a model-specific template
        return format_messages_fallback(messages)

def evaluate_presupposition(model, tokenizer, presupposition, model_name):
    """
    Evaluate a presupposition and return True/False response
    """
    # Generate chat messages
    messages = get_chat_messages(presupposition, model_name)
    
    # Set up generation parameters
    gen_kwargs = {
        "max_new_tokens": 64,  # Shorter response needed for True/False
        "temperature": 0.0,
        "top_p": 0.9,
        "do_sample": False
    }
    
    # Adjust parameters based on model type
    model_name_lower = model_name.lower()
    if any(name in model_name_lower for name in ["llama-2", "llama-3", "mistral-instruct"]):
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    
    # Create pipeline for generation
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **gen_kwargs
    )
    
    # Apply the chat template to get a formatted prompt string
    prompt = apply_template(messages, tokenizer, model_name)
    
    # Generate response
    output = generator(prompt, return_full_text=False)[0]['generated_text']
    response = output.strip()
    
    # Extract True/False answer (model might include additional text)
    if "true" in response.lower():
        return "True"
    elif "false" in response.lower():
        return "False"
    else:
        # If the model doesn't respond with a clear True/False, we'll log and return the raw response
        print(f"Unclear response for '{presupposition}': {response}")
        return response
    
def load_presuppositions(filepath):
    """
    Load presuppositions from a text file
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        presuppositions = [line.strip() for line in file if line.strip()]
    return presuppositions

def find_eligible_questions(results_file):
    """
    Find questions with Response_1 = 0 in the results file
    """
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return []
    
    try:
        # Read the results file
        results_df = pd.read_csv(results_file, index_col=0)
        
        # Check if Response_1 column exists
        if 'Response_1' not in results_df.columns:
            print(f"Response_1 column not found in {results_file}")
            return []
        
        # Get questions with Response_1 = 0
        eligible_questions = results_df[results_df['Response_1'] == 0].index.tolist()
        
        print(f"Found {len(eligible_questions)} questions with Response_1 = 0")
        return eligible_questions
    
    except Exception as e:
        print(f"Error processing results file: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Evaluate presuppositions using LLM")
    parser.add_argument("model_name", type=str, help="Hugging Face model name")
    parser.add_argument("--presuppositions_file", type=str, default="data/presuppositions.txt", 
                       help="Path to file containing presuppositions")
    parser.add_argument("--batch_size", type=int, default=10, 
                       help="Number of presuppositions to process in each batch")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Custom output directory")
    args = parser.parse_args()
    
    model_name = args.model_name
    presuppositions_file = args.presuppositions_file
    batch_size = args.batch_size
    
    # Create a shorter model identifier for directory names
    model_id = model_name.split("/")[-1]
    
    # Ensure the output directory exists
    output_dir = args.output_dir if args.output_dir else f"results/{model_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output files
    output_file = f"{output_dir}/presupposition_evaluation.csv"
    factual_check_file = f"{output_dir}/factual_check.txt"
    
    # Define results file to check
    results_file = f"{output_dir}/prompt0_result_all.csv"
    
    # Log information about the run
    print(f"Processing model: {model_name}")
    print(f"Presuppositions file: {presuppositions_file}")
    print(f"Results file to check: {results_file}")
    print(f"Output file: {output_file}")
    print(f"Factual check summary: {factual_check_file}")
    print(f"Batch size: {batch_size}")
    
    # Find questions with Response_1 = 0
    eligible_questions = find_eligible_questions(results_file)
    
    if not eligible_questions:
        print("No eligible questions found. Exiting.")
        return
    
    # Load presuppositions
    presuppositions = load_presuppositions(presuppositions_file)
    print(f"Loaded {len(presuppositions)} presuppositions from file")
    
    # Load the model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Check if the model has a chat template and log
    has_chat_template = hasattr(tokenizer, 'apply_chat_template')
    if has_chat_template:
        print("Model has a built-in chat template: Yes")
    else:
        print("Model has a built-in chat template: No (using fallback)")
    
    # Initialize CSV file with header if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Question', 'Presupposition', 'Evaluation'])
    
    # Track counts of True/False responses
    true_count = 0
    false_count = 0
    other_count = 0  # For responses that aren't clearly True or False
    
    # Process presuppositions in batches
    for i, presupposition in enumerate(tqdm(presuppositions, desc="Processing presuppositions")):
        # Skip if the index doesn't match any eligible question
        if i not in range(len(eligible_questions)):
            continue
            
        try:
            # Evaluate the presupposition
            evaluation = evaluate_presupposition(model, tokenizer, presupposition, model_name)
            
            # Update counts
            if evaluation == "True":
                true_count += 1
            elif evaluation == "False":
                false_count += 1
            else:
                other_count += 1
                
            # Append result to CSV
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([i, presupposition, evaluation])
        except Exception as e:
            print(f"Error evaluating presupposition '{presupposition}': {e}")
            continue
    
    # Calculate total and percentages
    total_evaluated = true_count + false_count + other_count
    true_percentage = (true_count / total_evaluated * 100) if total_evaluated > 0 else 0
    false_percentage = (false_count / total_evaluated * 100) if total_evaluated > 0 else 0
    other_percentage = (other_count / total_evaluated * 100) if total_evaluated > 0 else 0
    
    # Write the summary to the factual check file
    with open(factual_check_file, 'w', encoding='utf-8') as f:
        f.write(f"Factual Check Summary for {model_name}\n")
        f.write("=======================================\n\n")
        f.write(f"Total presuppositions evaluated: {total_evaluated}\n\n")
        f.write(f"True responses: {true_count} ({true_percentage:.2f}%)\n")
        f.write(f"False responses: {false_count} ({false_percentage:.2f}%)\n")
        f.write(f"Other responses: {other_count} ({other_percentage:.2f}%)\n\n")
        
        # Add accuracy assessment (assuming all presuppositions are false)
        accuracy = (false_count / total_evaluated * 100) if total_evaluated > 0 else 0
        f.write(f"Accuracy (assuming all presuppositions are false): {accuracy:.2f}%\n")
    
    print(f"\nCompleted evaluation of {total_evaluated} presuppositions")
    print(f"True responses: {true_count} ({true_percentage:.2f}%)")
    print(f"False responses: {false_count} ({false_percentage:.2f}%)")
    print(f"Other responses: {other_count} ({other_percentage:.2f}%)")
    print(f"Final results saved to {output_file}")
    print(f"Summary saved to {factual_check_file}")

if __name__ == "__main__":
    main()