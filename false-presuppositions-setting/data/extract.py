import json

def process_files():
    # Read the questions file
    with open('questions.txt', 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Read the jsonl file
    data = []
    with open('test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Ensure we have the same number of questions and data entries
    if len(questions) != len(data):
        print(f"Warning: Number of questions ({len(questions)}) doesn't match number of entries in jsonl ({len(data)})")
    
    # Extract presuppositions[0] and corrections[0] for each question
    presuppositions = []
    corrections = []
    
    for i, question in enumerate(questions):
        # Find the corresponding entry in data
        for entry in data:
            if entry.get('question') == question:
                # Extract first presupposition and correction
                if 'presuppositions' in entry and len(entry['presuppositions']) > 0:
                    presuppositions.append(entry['presuppositions'][0])
                else:
                    presuppositions.append("")  # Add empty string if no presupposition exists
                
                if 'corrections' in entry and len(entry['corrections']) > 0:
                    corrections.append(entry['corrections'][0])
                else:
                    corrections.append("")  # Add empty string if no correction exists
                
                break
        else:
            # If question not found in data
            print(f"Warning: Question not found in jsonl: '{question}'")
            presuppositions.append("")
            corrections.append("")
    
    # Write the results to output files
    with open('presuppositions.txt', 'w', encoding='utf-8') as f:
        for presupposition in presuppositions:
            f.write(presupposition + '\n')
    
    with open('corrections.txt', 'w', encoding='utf-8') as f:
        for correction in corrections:
            f.write(correction + '\n')
    
    print(f"Processed {len(questions)} questions")
    print(f"Wrote {len(presuppositions)} presuppositions to presuppositions.txt")
    print(f"Wrote {len(corrections)} corrections to corrections.txt")

if __name__ == "__main__":
    process_files()