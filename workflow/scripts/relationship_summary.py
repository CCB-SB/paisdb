import pandas as pd
import numpy as np
import os

'''
This file was used to produce the tables in the Appendix of the thesis
it creates laTex ready longtable environments
'''

methods = ["Mistral-Small-Instruct-2409", "BioMistral-7B"]
precisions = [0.69, 0.36]
recalls = [0.82, 0.97]
f1_scores = [0.80, 0.35]
balanced_accuracys = [0.81, 0.53]



for file_path in list(snakemake.input.files):
    reference = pd.read_csv(file_path)
    file_name = os.path.basename(file_path).replace('_', ' ')[:-4]

    methods.append(file_name)

    classifications = {
        word: reference['classification'].str.count(word).sum()
        for word in ['TP', 'FP', 'TN', 'FN']
    }

    precision = classifications['TP'] / (classifications['TP'] + classifications['FP']) if (classifications['TP'] + classifications['FP']) != 0 else 0
    recall = classifications['TP'] / (classifications['TP'] + classifications['FN']) if (classifications['TP'] + classifications['FN']) != 0 else 0
    true_positive_rate = classifications['TP'] / (classifications['TP'] + classifications['FN'])
    true_negative_rate = classifications['TN'] / (classifications['TN'] + classifications['FP'])
    
    precisions.append(precision)
    recalls.append(recall)
    balanced_accuracys.append((true_positive_rate + true_negative_rate) / 2)
    f1_scores.append(2* precision * recall / (precision + recall) if (precision + recall) != 0 else 0)

outpath = f"{snakemake.output.txt}"

# write everything into latex longtable format
with open(outpath, 'w') as f:
    f.write("\hline\n")
    f.write("\\textbf{method} & \\textbf{precision} & \\textbf{recall} & \\textbf{f1-score} & \\textbf{balanced accuracy}\\\\\n")
    f.write("\endhead\n")

    f.write("\hline\n")
    for name, precision, recall, f1_score, balanced_accuracy in zip(methods, precisions, recalls, f1_scores, balanced_accuracys):
        f.write(f"{name} & {precision:.3f} & {recall:.3f} & {f1_score:.3f} & {balanced_accuracy:.3f}\\\\ \n")
        f.write("\hline\n")
        

        