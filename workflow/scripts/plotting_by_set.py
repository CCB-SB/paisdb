import sys
from os.path  import abspath, basename
sys.path.insert(0, abspath("scripts"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from utils.color_mapper import colormapper24

# calculate precision, recall and f1-score
def calc_metric(source):
    TP = sum(source['TP'])
    FP = sum(source['FP'])
    FN = sum(source['FN'])

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

rows = {}

inputs = snakemake.input.files

# collect the evaluations from the results of the testers
for file_path in list(inputs):
    reference = pd.read_csv(file_path)

    file_name = str(basename(file_path))[:-4]

    precisions = []
    recalls = []
    f1_scores = []

    # total
    precision, recall, f1_score = calc_metric(reference)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

    # Abstracts only
    abstracts = reference[reference['pmc'].isna()]
    precision, recall, f1_score = calc_metric(abstracts)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

    # full article only
    full_article = reference[reference['pmc'].notna()]
    precision, recall, f1_score = calc_metric(full_article)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

    # empty reference
    empty_ref = reference[reference['reference'].isna()]
    precision, recall, f1_score = calc_metric(empty_ref)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

    # atleast one reference
    atleast_one_ref = reference[reference['reference'].notna()]
    precision, recall, f1_score = calc_metric(atleast_one_ref)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)

    rows[file_name] = (precisions, recalls, f1_scores)

#################################################
## Create the plot
#################################################
ColorMapper = colormapper24()

color_labels = list(rows.keys())
colors = [ColorMapper.map(approach) for approach in color_labels]
color_labels = [textwrap.fill(label.replace('_', ' ',), width=20) for label in color_labels]
assert 'white' not in colors, f"Some label was not converted: {[approach for approach in color_labels if ColorMapper.map(approach) == 'white']}"

# group labels
x_labels = ['Total', 'Abstracts', 'Full\narticle', 'Empty', 'Non\nEmpty']

# group values
values = [f1_score for _, _ , f1_score in rows.values()]
#values = np.array(values).T

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

plt.ylabel('f1-score')
plt.xlabel('data subset')
plt.title(f'Comparison of {str(snakemake.params.name)} extraction f1-scores ')

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

    



    


