import pandas as pd
import numpy as np
import os

'''
This file was used to produce the tables in the Appendix of the thesis
it creates laTex ready longtable environments
'''

def calc_metric(source):
    TP = sum(source['TP'])
    FP = sum(source['FP'])
    FN = sum(source['FN'])

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    if 'illformed' in source.columns:
        illformed = sum(source['illformed'])
    else:
        illformed = None


    return [precision, recall, f1_score, illformed]

# score collections
subset_sizes = []
total_scores = []
abstract_scores = []
full_article_scores = []
empty_scores = []
non_empty_scores = []


outpath = str(snakemake.output.txt)


# add scores of subsets for each file
for file_path in list(snakemake.input.files):
    reference = pd.read_csv(file_path)

    file_name = os.path.basename(file_path).replace('_', ' ')[:-4]

    # total
    total_scores.append([file_name] + calc_metric(reference))
    subset_sizes.append(len(reference))

    # Abstracts only
    abstracts = reference[reference['pmc'].isna()]
    abstract_scores.append([file_name] + calc_metric(abstracts))
    subset_sizes.append(len(abstracts))

    # full article only
    full_article = reference[reference['pmc'].notna()]
    full_article_scores.append([file_name] + calc_metric(full_article))
    subset_sizes.append(len(full_article))

    # empty reference
    empty_ref = reference[reference['reference'].isna()]
    empty_scores.append([file_name] + calc_metric(empty_ref))
    subset_sizes.append(len(empty_ref))
    
    # atleast one reference
    atleast_one_ref = reference[reference['reference'].notna()]
    non_empty_scores.append([file_name] + calc_metric(atleast_one_ref))
    subset_sizes.append(len(atleast_one_ref))

# write everything into latex longtable format
def segment_printer(f, name, subset_size, scores):
    f.write("\\begin{longtable}{|c|c|c|c|c|}\n")
    f.write("\t\hline\n")
    f.write("\t & \multicolumn{4}{c|}{\\textbf{"+ name + " (" + str(subset_size) + ")}}\\\\\n")
    f.write("\t\hline\n")
    f.write("\t \\textbf{method} & \\textbf{precision} & \\textbf{recall} & \\textbf{f1-score} & \\textbf{invalid answers}\\\\\n")
    f.write("\t\hline\n")
    f.write("\t\endhead\n")
    for approach, precision, recall , f1_score, invalid in scores:
        invalid_format = f"{str(invalid)} ({(invalid / subset_size):.1%})".replace("%", "\%") if invalid is not None else '-'
        f.write(f"\t{approach} & {precision:.3f} & {recall:.3f} & {f1_score:.3f} & {invalid_format}\\\\\n")
        f.write("\t\hline\n")
    f.write("\\end{longtable}\n")
    

with open(outpath, 'w') as f:
    segments = [("total", total_scores), ("abstracts", abstract_scores), ("full article", full_article_scores), ("empty", empty_scores), ("non-empty", non_empty_scores)]
    for index, (name, source) in enumerate(segments):
        segment_printer(f, name, subset_sizes[index], source)

