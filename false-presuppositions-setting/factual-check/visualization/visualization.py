import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data setup
data = {
    "Model": [
        "Gemma-2-9b-it",
        "LLaMA-3.1-8B-Instruct",
        "LLaMA-3.3-70B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-72B-Instruct"
    ],
    "Total": [72, 104, 83, 86, 72, 72],
    "True": [26, 26, 29, 23, 15, 21],
    "False": [42, 78, 54, 61, 37, 40],
    "Not Sure": [4, 0, 0, 2, 20, 11],
    "Family": [
        "Gemma", "LLaMA", "LLaMA",
        "Qwen", "Qwen", "Qwen"
    ]
}

df = pd.DataFrame(data)

# Compute percentages
df_percent = df.copy()
for col in ["True", "False", "Not Sure"]:
    df_percent[col] = df[col] / df["Total"] * 100

# Model sizes (corrected)
model_labels = ["9B", "8B", "70B", "7B", "14B", "72B"]

# Grouping positions
family_positions = {
    "Gemma": [i for i, fam in enumerate(df["Family"]) if fam == "Gemma"],
    "LLaMA": [i for i, fam in enumerate(df["Family"]) if fam == "LLaMA"],
    "Qwen": [i for i, fam in enumerate(df["Family"]) if fam == "Qwen"]
}
family_midpoints = {fam: np.mean(positions) for fam, positions in family_positions.items()}

# Values
false_vals = df_percent["False"]
true_vals = df_percent["True"]
ns_vals = df_percent["Not Sure"]

# Color scheme
colors = {
    "False": "#3182bd",    # Blue for correct
    "True": "#999999",     # Gray for incorrect
    "Not Sure": "#bdbdbd"  # Light gray
}

# Plot
# Raise model size labels by 2 (from -11.5 → -9.5)
fig, ax = plt.subplots(figsize=(6.5, 4.8))
x = range(len(df))
bar_width = 0.6
half_width = bar_width / 2 - 0.03

# Stacked bars
ax.bar(x, false_vals, width=bar_width, label='False (Correct)', color=colors["False"])
ax.bar(x, true_vals, width=bar_width, bottom=false_vals, label='True (Incorrect)', color=colors["True"])
ax.bar(x, ns_vals, width=bar_width, bottom=false_vals + true_vals, label='Not Sure', color=colors["Not Sure"])

# Percentage annotations
for i in range(len(df)):
    f, t, n = false_vals[i], true_vals[i], ns_vals[i]
    if f > 0:
        ax.text(i, f / 2, f'{f:.0f}%', ha='center', va='center', color='white', fontsize=21)
    if t > 0:
        ax.text(i, f + t / 2, f'{t:.0f}%', ha='center', va='center', color='white', fontsize=21)
    if n > 0:
        ax.text(i, f + t + n / 2, f'{n:.0f}%', ha='center', va='center', color='black', fontsize=21)

# x-axis hidden
ax.set_xticks(x)
ax.set_xticklabels([""] * len(x))

# Group labels and lines
line_y = -5
label_y = -14
for fam, positions in family_positions.items():
    start = min(positions) - half_width
    end = max(positions) + half_width
    mid = family_midpoints[fam]
    ax.hlines(line_y, start, end, colors='gray', linewidth=1.5)
    ax.text(mid, label_y, fam, ha='center', va='top', fontsize=20, fontweight='bold', color='black')

# Raise model size labels to -9.5
for i, label in enumerate(model_labels):
    ax.text(i, -9.5, label, ha='center', va='top', fontsize=13, color='black')

# Axis & grid
ax.set_ylim(-18, 98)
# ax.set_ylabel('Response Distribution (%)', fontsize=
ax.grid(False)

# Remove -20 y-label
ticks = ax.get_yticks()
tick_labels = ["" if int(t) == -20 else str(int(t)) for t in ticks]
ax.set_yticklabels(tick_labels)

# Legend (still without title, raised to -0.04)
legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=3,
    frameon=True,
    fancybox=False,
    edgecolor='black',
    fontsize=10
)
# legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.show()
