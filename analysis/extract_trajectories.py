import os
import pandas as pd

def extract_trajectories_to_csv():
    base_dir = "results/Meta-Llama-3.1-8B-Instruct"
    output_dir = "output/Meta-Llama-3.1-8B-Instruct"

    pos_responses = []
    neg_responses = []

    for i in range(1, 5):  # prompt1 to prompt4
        result_path = os.path.join(base_dir, f"prompt{i}_results.txt")
        csv_path = os.path.join(output_dir, f"prompt{i}.csv")

        with open(result_path, "r") as f:
            flip_turns = [int(line.strip().split(":")[-1]) for line in f if line.strip()]

        df = pd.read_csv(csv_path)

        for idx, turn in enumerate(flip_turns):
            if idx >= len(df):
                continue

            # Positive response: from response_{turn}, only if 1 <= turn <= 5
            if 1 <= turn <= 5:
                col = f"Response_{turn}"
                if col in df.columns and pd.notna(df.at[idx, col]):
                    pos_responses.append({"response": df.at[idx, col].strip()})

            # Negative response: from response_{turn+1}, only if 0 <= turn < 5
            if 0 <= turn < 5:
                col = f"Response_{turn + 1}"
                if col in df.columns and pd.notna(df.at[idx, col]):
                    neg_responses.append({"response": df.at[idx, col].strip()})

    # Save to CSV
    pd.DataFrame(pos_responses).to_csv("analysis/data/positive_trajectory.csv", index=False)
    pd.DataFrame(neg_responses).to_csv("analysis/data/negative_trajectory.csv", index=False)

    print(f"✅ Saved {len(pos_responses)} positive and {len(neg_responses)} negative responses to CSV.")

if __name__ == "__main__":
    extract_trajectories_to_csv()
