#!/usr/bin/env python3
"""
This script reads a text file line by line, extracts the last character from each line,
converts it to an integer, and calculates the mean of those integers.

Usage:
    python calculate_mean.py --input FILE_PATH
"""

import argparse
import statistics
from typing import List


def calculate_mean_from_file(file_path: str) -> float:
    """
    Read a file line by line, extract the last character of each line,
    convert to integers, and calculate their mean.
    
    Args:
        file_path (str): Path to the input text file
    
    Returns:
        float: The mean value of the last characters as integers
    """
    values = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, 1):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Extract the last character
            last_char = line.strip()[-1]
            
            try:
                # Convert to integer
                value = int(last_char)
                values.append(value)
                
                # Print progress every 50 lines
                if i % 50 == 0:
                    print(f"Processed {i} lines...")
                    
            except ValueError:
                print(f"Warning: Line {i} has a non-integer last character: '{last_char}'")
    
    # Calculate and return the mean
    if values:
        return statistics.mean(values)
    else:
        return 0.0


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Calculate the mean of the last character of each line in a file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input text file')
    
    args = parser.parse_args()
    
    try:
        # Calculate the mean
        mean_value = calculate_mean_from_file(args.input)
        
        # Print the results
        print(f"\nResults:")
        print(f"Mean of the last characters: {mean_value:.2f}")
        
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()