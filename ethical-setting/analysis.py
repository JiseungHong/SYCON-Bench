# compare_reasoner_vs_chat_by_bias_type.py

import csv
from collections import defaultdict

# File paths
csv_path = "data/stereoset_intra_user_queries_api_over45.csv"
chat_path = "results/gpt-4o/prompt0_results.txt"
reasoner_path = "results/o3-mini/prompt0_results.txt"

# Load bias_type from the CSV
bias_types = []
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        bias_types.append(row["bias_type"])

assert len(bias_types) == 200, "Expected 200 rows in CSV"

# Load scores from result files
with open(chat_path, encoding='utf-8') as f:
    chat_scores = [int(line.strip()[-1]) for line in f.readlines()]
with open(reasoner_path, encoding='utf-8') as f:
    reasoner_scores = [int(line.strip()[-1]) for line in f.readlines()]

assert len(chat_scores) == 200 and len(reasoner_scores) == 200, "Expected 200 lines in each results file"

# Initialize stats per bias type
bias_summary = defaultdict(lambda: {
    "reasoner_better": 0,
    "chat_better": 0,
    "tie": 0,
    "chat_better_indices": []
})

# Compare scores
for i in range(200):
    bias = bias_types[i]
    chat = chat_scores[i]
    reasoner = reasoner_scores[i]

    if reasoner > chat:
        bias_summary[bias]["reasoner_better"] += 1
    elif chat > reasoner:
        bias_summary[bias]["chat_better"] += 1
        bias_summary[bias]["chat_better_indices"].append(i)
    else:
        bias_summary[bias]["tie"] += 1

# Print summary
print("=== Comparison by Bias Type ===")
print(f"{'Bias Type':<20} | Reasoner > Chat | Chat > Reasoner | Tie")
print("-" * 60)
for bias, stats in bias_summary.items():
    print(f"{bias:<20} | {stats['reasoner_better']:^15} | {stats['chat_better']:^16} | {stats['tie']:^3}")

# Print indices where chat > reasoner
print("\n=== Chat > Reasoner Question Indices by Bias Type ===")
for bias, stats in bias_summary.items():
    if stats["chat_better_indices"]:
        print(f"{bias}: {stats['chat_better_indices']}")
