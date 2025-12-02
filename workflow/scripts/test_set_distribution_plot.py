import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap

# ensure result path is available
if not os.path.exists("../results/images"):
    os.makedirs("../results/images") 

dataset = pd.read_csv("../source/test_data_refined_2.csv")
dataset.replace({np.nan:None}, inplace=True)


# generate count plots for interesting fields
fig, axes = plt.subplots(2, 3, constrained_layout=True)
plot_assignment = [
    ("strains", "strains", axes[0,0]), ("pathogen validation methode", "validation", axes[0,1]), ("cohort size", "cohort", axes[0,2]),
    ("target organs", "anatomy_manual", axes[1,0]), ("symptoms","symptoms_manual", axes[1,1])
]
axes[1,2].set_visible(False)

for name, column, axis in plot_assignment:
    abstracts = dataset[dataset['pmc'].isna()]
    full_article = dataset[dataset['pmc'].notna()]
    
    # transform string representation of list to lenght of counts
    abstract_lengths = abstracts[column].apply(lambda x : len(x.split(',')) if x else 0)
    abstract_counts = abstract_lengths.value_counts().sort_index()

    full_article_lengths = full_article[column].apply(lambda x : len(x.split(',')) if x else 0)
    full_article_counts = full_article_lengths.value_counts().sort_index()

    # create a dataframe to use pd plot functions
    counts = pd.DataFrame({
        'Abstracts': abstract_counts,
        'Full articles': full_article_counts
    }).fillna(0).astype(int)

    counts.plot(kind='bar', stacked=True, color=['blue', 'orange'], 
        ax=axis, xlabel=name, legend=False)
    
fig.legend(['Abstract', 'Full article'], loc='lower right', title="Article type",
    ncol=1, bbox_to_anchor=(0.95, 0.25))

fig.suptitle("Distribution of manual extractions per modul")

plt.savefig(f'../results/images/distributionExtractions.png', dpi=300, bbox_inches='tight')
plt.close()  

