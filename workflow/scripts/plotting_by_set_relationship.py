import sys
from os.path  import abspath, basename
sys.path.insert(0, abspath("scripts"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from utils.color_mapper import colormapper24

# retrieve number for each classification and
# apply metrics
def calc_metric(source):
    TP = len(source[source['classification'] == 'TP'])
    FP = len(source[source['classification'] == 'FP'])
    FN = len(source[source['classification'] == 'FN'])
    TN = len(source[source['classification'] == 'TN'])

    assert len(source) == TP + FP + FN + TN, "Missing values in statistic extraction"

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    true_positive_rate = TP / (TP + FN)
    true_negative_rate = TN / (TN + FP)

    balanced_accuracy = (true_positive_rate + true_negative_rate) / 2

    return [precision, recall, f1_score, balanced_accuracy]

rows = {}

inputs = snakemake.input.files

# extract classifications from the tester results
for file_path in list(inputs):
    reference = pd.read_csv(file_path)

    file_name = str(basename(file_path))[:-4]

    rows[file_name] = calc_metric(reference)

# add llm results from "Benchmarking Large Language Models for Pathogen–Disease Classification in Post-Acute Infection Syndromes"
rows['Mistral-Small-Instruct-2409'] = [0.69, 0.82, 0.80, 0.81]
rows['BioMistral-7B'] = [0.36, 0.97, 0.35, 0.53]

ColorMapper = colormapper24()

color_labels = list(rows.keys())
colors = [ColorMapper.map(approach) for approach in color_labels]
color_labels = [textwrap.fill(label.replace('_', ' ',), width=20) for label in color_labels]
assert 'white' not in colors, f"Some label was not converted: {[approach for approach in color_labels if ColorMapper.map(approach) == 'white']}"

# group labels
x_labels = ['precision', 'recall', 'f1-score', 'balanced\naccuracy']

# group values
values = np.array(list(rows.values()))

print(values)

# Width of each bar
bar_width = 0.2

# bars per group
bars_per_group = len(color_labels)

# X positions for each group
group_offset = (len(color_labels) + 2) * bar_width
x = np.arange(len(x_labels)) * group_offset

# Plot each bar type (so same color repeats across groups)
for i in range(bars_per_group):
    plt.bar(
        x + (i - (bars_per_group - 1)/2) * bar_width,  
        values[i],
        width=bar_width,
        color=colors[i],
        label=color_labels[i]
    )

plt.xticks(x, x_labels)

plt.ylabel('Metric score')
plt.title('Comparison of approach metrics')

plt.legend(
    title="Approaches",
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)
plt.tight_layout()
plt.grid(True,
    linewidth=0.5,
    alpha=0.7,
    color='black',
    axis='y',
    )


plt.savefig(str(snakemake.output.png), dpi=300, bbox_inches='tight')

    



    


