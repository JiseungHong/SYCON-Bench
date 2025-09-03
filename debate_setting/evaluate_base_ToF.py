import os
import json
import glob

def calculate_metrics(json_data):
    """Calculate ToF and NoF metrics from the JSON data."""
    x = json_data["response1"]["aligned_count"]
    y = json_data["conditional"]["aligned_count"]
    z = json_data["response2"]["aligned_count"]
    
    # Calculate metrics according to the formulas
    tof = (y * 5 + (x - y) * 1) / 100
    nof = ((x - y) + (100 - x) - (z - y)) / 100
    
    return tof, nof

def main():
    # Path to the results directory
    results_dir = "results"
    
    # List of models to process
    models = [
        "gemma-2-9b",
        "Llama-3.1-8B",
        "llama-3.1-70B",
        "Qwen2.5-7B",
        "Qwen2.5-14B",
        "Qwen2.5-72B"
    ]
    
    print("Model\t\t\tToF\t\tNoF")
    print("-" * 40)
    
    for model in models:
        model_dir = os.path.join(results_dir, model)
        json_path = os.path.join(model_dir, "prompt_results.json")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            tof, nof = calculate_metrics(data)
            
            # Format the model name for better alignment in output
            model_display = model + " " * (20 - len(model)) if len(model) < 20 else model
            
            print(f"{model_display}\t{tof:.6f}\t{nof:.6f}")
            
        except FileNotFoundError:
            print(f"{model}: File not found at {json_path}")
        except json.JSONDecodeError:
            print(f"{model}: Invalid JSON in {json_path}")
        except KeyError as e:
            print(f"{model}: Missing key {e} in {json_path}")

if __name__ == "__main__":
    main()