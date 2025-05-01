#!/usr/bin/env python3
"""
false_presupposition_question_extractor.py

This script extracts questions that contain false presuppositions from a JSONL file.
It processes test.jsonl and extracts up to 200 randomly selected questions to questions.txt,
their corresponding presuppositions to presuppositions.json, and corrections to corrections.json
when both strings in "raw_labels" are "false presupposition".
"""

import json
import sys
import random

def extract_false_presupposition_questions(
    input_file, 
    questions_file, 
    presuppositions_file, 
    corrections_file, 
    max_questions=200
):
    """
    Extract up to max_questions randomly selected questions with false presuppositions from the input JSONL file.
    
    Args:
        input_file (str): Path to the input JSONL file
        questions_file (str): Path to output file for questions
        presuppositions_file (str): Path to output JSON file for presuppositions
        corrections_file (str): Path to output JSON file for corrections
        max_questions (int): Maximum number of questions to extract
    
    Raises:
        ValueError: When both strings in "raw_labels" are "false presupposition" but "presuppositions" is empty
    """
    # First pass: collect all eligible entries
    eligible_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(f_in, 1):
            try:
                # Parse the JSON line
                data = json.loads(line.strip())
                
                # Extract the fields
                question = data.get("question")
                raw_labels = data.get("raw_labels", [])
                presuppositions = data.get("presuppositions", [])
                corrections = data.get("corrections", [])
                
                # Check if the question contains false presupposition (both labels are "false presupposition")
                if len(raw_labels) == 2 and all(label == "false presupposition" for label in raw_labels):
                    # Raise error if presuppositions is empty
                    if not presuppositions:
                        raise ValueError(f"Error on line {line_num}: Both strings in 'raw_labels' are 'false presupposition' but 'presuppositions' is empty")
                    
                    # Add to eligible entries
                    eligible_entries.append({
                        "question": question,
                        "presuppositions": presuppositions,
                        "corrections": corrections if corrections else []
                    })
            
            except json.JSONDecodeError:
                print(f"Error: Failed to parse JSON on line {line_num}", file=sys.stderr)
            except KeyError as e:
                print(f"Error: Missing required field {e} on line {line_num}", file=sys.stderr)
            except Exception as e:
                print(f"Error on line {line_num}: {str(e)}", file=sys.stderr)
                raise
    
    # Select random subset if we have more than max_questions
    if len(eligible_entries) > max_questions:
        selected_entries = random.sample(eligible_entries, max_questions)
    else:
        selected_entries = eligible_entries
    
    # Write selected questions to the text file
    with open(questions_file, 'w', encoding='utf-8') as f_questions:
        for entry in selected_entries:
            f_questions.write(f"{entry['question']}\n")
    
    # Write presuppositions to the JSON file (each line is a JSON array)
    with open(presuppositions_file, 'w', encoding='utf-8') as f_presuppositions:
        for entry in selected_entries:
            f_presuppositions.write(json.dumps(entry['presuppositions']) + "\n")
    
    # Write corrections to the JSON file (each line is a JSON array)
    with open(corrections_file, 'w', encoding='utf-8') as f_corrections:
        for entry in selected_entries:
            f_corrections.write(json.dumps(entry['corrections']) + "\n")

    return len(selected_entries)

def main():
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    input_file = "test.jsonl"
    questions_file = "questions.txt"
    presuppositions_file = "presuppositions.json"
    corrections_file = "corrections.json"
    max_questions = 200
    
    try:
        num_extracted = extract_false_presupposition_questions(
            input_file, questions_file, presuppositions_file, corrections_file, max_questions
        )
        print(f"Successfully extracted {num_extracted} out of {max_questions} requested false presupposition questions")
        print(f"- Questions saved to: {questions_file}")
        print(f"- Presuppositions saved to: {presuppositions_file}")
        print(f"- Corrections saved to: {corrections_file}")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()