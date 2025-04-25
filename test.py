import matplotlib.pyplot as plt
import numpy as np

models = ['Pythia', 'Pythia-PANDA', 'Mamba-hf', 'Mamba-PANDA']
categories = ['Gender', 'Profession', 'Race', 'Religion']

# Score data for each metric
ppl_data = {
    'Gender': [125.3, 355.3, 117.5, 108.9],
    'Profession': [109.4, 274.6, 111.5, 107.3],
    'Race': [102.7, 258.3, 100.9, 97.6],
    'Religion': [145.5, 484.7, 166.5, 163.1]
}

lm_data = {
    'Gender': [93.5, 92.4, 92.6, 93.3],
    'Profession': [91.7, 91.1, 92.3, 92.3],
    'Race': [91.7, 91.6, 92.6, 92.8],
    'Religion': [95.4, 94.5, 94.3, 94.7]
}

ss_data = {
    'Gender': [70.6, 73.6, 63.4, 62.5],
    'Profession': [74.4, 77.0, 72.3, 72.7],
    'Race': [76.8, 76.9, 70.9, 70.9],
    'Religion': [67.2, 74.5, 69.3, 70.2]
}

icat_data = {
    'Gender': [66.0, 68.0, 58.7, 58.3],
    'Profession': [68.2, 70.1, 66.7, 67.1],
    'Race': [70.5, 70.4, 65.6, 65.8],
    'Religion': [64.1, 70.4, 63.3, 66.5]
}

fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)

data_list = [
    (ppl_data, 'Perplexity (PPL)', '', (90, 500), 0.5),
    (lm_data, 'LM Score', '', (90, 100), 0.5),
    (ss_data, 'Stereotype Score (SS)', '', (60, 80), 1.0),
    (icat_data, 'ICAT Score', '', (55, 75), 0.5)
]

x = np.arange(len(models))
width = 0.2

for ax, (data, title, ylabel, ylim, alpha) in zip(axes, data_list):
    for i, category in enumerate(categories):
        ax.bar(x + i * width, data[category], width=width, label=category, alpha=alpha)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

axes[0].legend(loc='upper left')
plt.tight_layout()
plt.show()