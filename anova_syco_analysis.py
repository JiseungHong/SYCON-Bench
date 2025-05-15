import os
import re
import scipy.stats as stats
from collections import defaultdict

SETTINGS = [
    'debate-setting',
    'ethical-setting',
    'false-presuppositions-setting'
]

PROMPT_FILES = [f'prompt{i}_results.txt' for i in range(5)]

def extract_tof_scores(filepath):
    scores = []
    with open(filepath, 'r') as f:
        for line in f:
            if 'ToF =' in line:
                match = re.search(r'ToF\s*=\s*(\d+)', line)
                if match:
                    scores.append(int(match.group(1)))
            else:
                stripped = line.strip()
                if stripped and stripped[-1].isdigit():
                    scores.append(int(stripped[-1]))
    return scores

def main():
    output_lines = []

    for setting in SETTINGS:
        output_lines.append(f"=== Setting: {setting} ===\n")
        base_dir = os.path.join(setting, 'results')
        model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        # --- Test 1: Across models using prompt0 ---
        prompt0_scores_by_model = {}
        for model in model_dirs:
            file_path = os.path.join(base_dir, model, 'prompt0_results.txt')
            if os.path.exists(file_path):
                prompt0_scores_by_model[model] = extract_tof_scores(file_path)

        output_lines.append("[ANOVA] Prompt 0: Across models\n")
        try:
            stat, pval = stats.f_oneway(*prompt0_scores_by_model.values())
            output_lines.append(f"F-statistic = {stat:.4f}, p-value = {pval:.4e}\n")
        except Exception as e:
            output_lines.append(f"ANOVA failed: {str(e)}\n")
        output_lines.append("\n")

        # --- Test 2: Within each model using prompts 1~4 ---
        for model in model_dirs:
            prompt_data = []
            for i in range(1, 5):
                path = os.path.join(base_dir, model, f'prompt{i}_results.txt')
                if os.path.exists(path):
                    prompt_data.append(extract_tof_scores(path))

            output_lines.append(f"[ANOVA] Prompts 1-4 within model: {model}\n")
            try:
                if all(len(p) == len(prompt_data[0]) for p in prompt_data):
                    stat, pval = stats.f_oneway(*prompt_data)
                    output_lines.append(f"F-statistic = {stat:.4f}, p-value = {pval:.4e}\n")
                else:
                    # Fallback to Kruskal-Wallis if uneven lengths
                    stat, pval = stats.kruskal(*prompt_data)
                    output_lines.append(f"(Kruskal-Wallis) H-statistic = {stat:.4f}, p-value = {pval:.4e}\n")
            except Exception as e:
                output_lines.append(f"Test failed: {str(e)}\n")
            output_lines.append("\n")

    # Save results to a file
    with open("consistency_anova_results.txt", "w") as out_file:
        out_file.writelines(output_lines)

if __name__ == "__main__":
    main()
