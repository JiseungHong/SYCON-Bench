import os
import re
import pandas as pd
from statsmodels.stats.anova import AnovaRM

def extract_values(filepath):
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            values.append(float(line.strip()[-1]))
    return values

def run_anova_for_model(model_path, model_name):
    prompt_files = [f"prompt{i}_results.txt" for i in range(1, 5)]
    data = []

    # Read prompt values
    for prompt in prompt_files:
        prompt_path = os.path.join(model_path, prompt)
        if os.path.exists(prompt_path):
            values = extract_values(prompt_path)
            for i, val in enumerate(values):
                data.append({
                    "subject": f"run_{i}",  # each run/row
                    "prompt": prompt.replace("_results.txt", ""),
                    "score": val
                })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Check that all subjects have data for all prompts
    n_prompts = len(df["prompt"].unique())
    counts = df.groupby("subject")["prompt"].nunique()
    valid_subjects = counts[counts == n_prompts].index
    df = df[df["subject"].isin(valid_subjects)]

    # Run repeated measures ANOVA
    result_path = os.path.join(model_path, "anova_results.txt")
    with open(result_path, "w") as f:
        if len(valid_subjects) < 2:
            f.write("Not enough valid paired data across prompts for ANOVA.\n")
        else:
            try:
                aovrm = AnovaRM(df, depvar="score", subject="subject", within=["prompt"])
                res = aovrm.fit()
                f.write(str(res))
            except Exception as e:
                f.write(f"Error running ANOVA: {str(e)}\n")

def main():
    base_dir = "results"
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if os.path.isdir(model_path):
            run_anova_for_model(model_path, model_name)

if __name__ == "__main__":
    main()
