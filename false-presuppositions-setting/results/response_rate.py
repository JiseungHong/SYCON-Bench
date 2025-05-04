import os
import pandas as pd
import glob
from collections import defaultdict

def analyze_directory(directory):
    """
    Analyze all prompt result files in a directory and create a response_rate.txt file
    showing counts of "1" values for each prompt and response column.
    """
    print(f"Analyzing directory: {directory}")
    
    # Find all CSV files matching the pattern prompt[0-4]_result_all.csv
    result_files = glob.glob(os.path.join(directory, "prompt*_result_all.csv"))
    
    if not result_files:
        print(f"No result files found in {directory}")
        return
    
    summary = []
    
    # Process each prompt file
    for file_path in sorted(result_files):
        try:
            # Extract prompt number from filename
            filename = os.path.basename(file_path)
            prompt_num = filename.split("_")[0]
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Count the number of "1"s in each response column
            response_counts = []
            for col in [col for col in df.columns if col.startswith("Response_")]:
                # Count occurrences of value 1
                count = (df[col] == 1).sum()
                response_counts.append(str(count))
            
            # Add to summary
            summary.append(f"{prompt_num}: {', '.join(response_counts)}")
            print(f"Processed {filename}: {', '.join(response_counts)}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write summary to response_rate.txt
    output_path = os.path.join(directory, "response_rate.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(summary))
    
    print(f"Created {output_path}")

def main():
    # Get all directories in the current directory
    base_dir = "."
    directories = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    # Filter out directories that likely don't contain model results
    exclude_dirs = [".git", "__pycache__", "data"]
    model_dirs = [d for d in directories if d not in exclude_dirs]
    
    if not model_dirs:
        print("No model directories found.")
        return
    
    print(f"Found {len(model_dirs)} model directories: {model_dirs}")
    
    # Process each model directory
    for directory in model_dirs:
        analyze_directory(os.path.join(base_dir, directory))
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()