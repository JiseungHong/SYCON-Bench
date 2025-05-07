import pandas as pd
import os

def find_missing_questions(file1_path, file2_path, output_path='missing_questions.csv'):
    """
    Compare questions between two CSV files and find which ones are missing in the second file.
    
    Parameters:
    - file1_path: Path to the first CSV file (stereoset_intra_user_queries_api_over45.csv)
    - file2_path: Path to the second CSV file (prompt0.csv)
    - output_path: Path where to save the missing questions (default: 'missing_questions.csv')
    
    Returns:
    - DataFrame containing missing questions and their row indices
    """
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # Print information about the files
        print(f"First file: {file1_path}")
        print(f"  - Shape: {df1.shape}")
        print(f"  - Columns: {df1.columns.tolist()}")
        
        print(f"\nSecond file: {file2_path}")
        print(f"  - Shape: {df2.shape}")
        print(f"  - Columns: {df2.columns.tolist()}")
        
        # Extract questions from the second file
        questions_file2 = set(df2['Question'].dropna().tolist())
        
        # Create a mask for missing questions
        missing_mask = ~df1['question'].isin(questions_file2)
        
        # Get the missing questions and their indices
        missing_df = df1[missing_mask].copy()
        
        # Print summary
        print(f"\nTotal questions in file1: {len(df1)}")
        print(f"Total questions in file2: {len(df2)}")
        print(f"Number of missing questions: {missing_mask.sum()}")
        
        # Save missing questions to a CSV file
        if not missing_df.empty:
            # Ensure the original row index is saved
            missing_df.reset_index(names=['original_row_index'], inplace=True)
            
            # Select only the necessary columns (original index and question)
            result_df = missing_df[['original_row_index', 'question']]
            
            # Rename for clarity
            result_df.columns = ['original_row_index', 'missing_question']
            
            # Save to CSV
            result_df.to_csv(output_path, index=False)
            print(f"\nMissing questions saved to: {output_path}")
            
            # Print a few examples
            print("\nSample of missing questions (with original row indices):")
            for _, row in result_df.head().iterrows():
                print(f"  - Row {row['original_row_index']}: {row['missing_question']}")
            
        return result_df
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    # Define file paths (adjust these according to your environment)
    stereoset_file = 'data/stereoset_intra_user_queries_api_over45.csv'
    prompt_file = 'output/claude-3-7-sonnet-20250219/prompt4.csv'
    output_file = 'missing_questions.csv'
    
    # Run the function
    missing_questions = find_missing_questions(stereoset_file, prompt_file, output_file)