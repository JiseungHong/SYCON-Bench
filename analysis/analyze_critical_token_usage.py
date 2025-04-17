import pandas as pd
from collections import defaultdict, Counter
import re

# === STEP 1: Parse contrastive_scores.txt ===
def get_top_critical_tokens(score_file, top_k=10):
    token_scores = defaultdict(list)

    with open(score_file, 'r') as f:
        current_example = []
        for line in f:
            if line.startswith("--- Example"):
                current_example = []
            elif line.strip():
                token, score = line.strip().split('\t')
                score = float(score)
                token = token.strip()
                if re.match(r"^\W*$", token):  # Skip pure punctuation
                    continue
                token_scores[token].append(score)

    # Compute average score and get top-k tokens with lowest average
    avg_scores = {tok: sum(scores)/len(scores) for tok, scores in token_scores.items()}
    top_critical = sorted(avg_scores.items(), key=lambda x: x[1])[:top_k]
    
    print("Top Critical:", top_critical)
    return [tok for tok, _ in top_critical]

# === STEP 2: Count token occurrences in each turn ===
def count_tokens_by_turn(df, tokens):
    token_counts = {tok: [0] * 5 for tok in tokens}

    for turn in range(1, 6):
        col = f"response_{turn}"
        for response in df[col].dropna():
            response_lower = response.lower()
            for tok in tokens:
                token_counts[tok][turn - 1] += response_lower.split().count(tok.lower())

    return token_counts

# === STEP 3: Main comparison logic ===
def compare_prompts(score_file, prompt1_path, prompt2_path, output_path="critical_token_stats.csv"):
    top_tokens = get_top_critical_tokens(score_file)
    print("Top Critical Tokens:", top_tokens)

    df1 = pd.read_csv(prompt1_path)
    df2 = pd.read_csv(prompt2_path)

    counts1 = count_tokens_by_turn(df1, top_tokens)
    counts2 = count_tokens_by_turn(df2, top_tokens)

    # Output: Rows = tokens, Columns = response_1 to response_5 (for each prompt)
    output = []
    for tok in top_tokens:
        row = {"token": tok}
        for i in range(5):
            row[f"prompt1_turn{i+1}"] = counts1[tok][i]
            row[f"prompt2_turn{i+1}"] = counts2[tok][i]
        output.append(row)

    out_df = pd.DataFrame(output)
    out_df.to_csv(output_path, index=False)
    print(f"Saved comparison to {output_path}")

if __name__ == "__main__":
    compare_prompts(
        score_file="contrastive_scores.txt",
        prompt1_path="results/Meta-Llama-3.1-8B-Instruct/prompt1.csv",
        prompt2_path="results/Meta-Llama-3.1-8B-Instruct/prompt2.csv"
    )
