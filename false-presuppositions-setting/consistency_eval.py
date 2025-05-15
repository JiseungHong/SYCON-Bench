import os
from itertools import combinations
from collections import defaultdict

# Directories containing different seeds of the same model
# seed_dirs = [
#     "results/Llama-3.1-8B-Instruct",
#     "results/Llama-3.1-8B-Instruct_seed307",
#     "results/Llama-3.1-8B-Instruct_seed411"
# ]

seed_dirs = [
    "results/Qwen2.5-14B-Instruct",
    "results/Qwen2.5-14B-Instruct_seed307",
    "results/Qwen2.5-14B-Instruct_seed411"
]

num_prompts = 5
file_prefix = "prompt"

def read_labels(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def compute_agreement(labels1, labels2):
    assert len(labels1) == len(labels2), "Mismatched file lengths."
    matches = sum([l1 == l2 for l1, l2 in zip(labels1, labels2)])
    return matches / len(labels1)

if __name__ == "__main__":
    print("=== Consistency Evaluation Across Seeds ===")
    
    for i in range(num_prompts):
        file_name = f"{file_prefix}{i}_results.txt"
        all_labels = []
        
        for path in seed_dirs:
            full_path = os.path.join(path, file_name)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Missing file: {full_path}")
            all_labels.append(read_labels(full_path))
        
        # Compute pairwise agreement across the three seeds
        agreements = []
        for idx1, idx2 in combinations(range(3), 2):
            score = compute_agreement(all_labels[idx1], all_labels[idx2])
            agreements.append(score)
        
        avg_agreement = sum(agreements) / len(agreements)
        print(f"{file_name}: Avg Pairwise Agreement = {avg_agreement:.4f}")
