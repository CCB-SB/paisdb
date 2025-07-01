import sys
from os.path import abspath
import pandas as pd
import os
import ast
import re
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
from utils_post_llm import process_result    
import numpy as np

import json

with open('abstracts_titles_dict_file.json', 'r') as file:
    data_dict = json.load(file)

dict_pmids = set(value for key, value in data_dict.items() if "pmid" in key)
print(dict_pmids)
def get_column_with_int(df):
  """
  Finds the name of the column in the DataFrame that contains 'int' in its name.

  Args:
    df: The pandas DataFrame to search.

  Returns:
    The name of the column containing 'int' in its name, or None if no such column exists.
  """
  for col in df.columns:
    if 'int' in col:
      return col
  return None



# print(str(snakemake.input.llms))
llm_directory = "/local/paisdb/results/relation_extraction/"
batch_size = 50
ground_truth_path = snakemake.input.csv
output_path = snakemake.output.csv
print(output_path)


# Load the ground truth file
ground_truth = pd.read_csv(ground_truth_path)

# Ensure ground truth contains a 'relationship' column
if "relationship" not in ground_truth.columns:
    raise ValueError("'relationships.csv' must contain a 'relationship' column.")

# Evaluation dictionary to store metrics
evaluation_dict = {}

df_combined = pd.DataFrame()
# Iterate over all CSV files in the directory


for file in os.listdir(llm_directory):
    if file.endswith(".csv"):
        file_path = os.path.join(llm_directory, file)
        print(file_path)
        df = pd.read_csv(file_path)
        print(len(df))
        df = df.drop(df.filter(regex='int').columns, axis=1)
        processed_df = process_result(df)
        processed_df = pd.concat([processed_df, df['time_taken']], axis=1)
        int_column = get_column_with_int(processed_df)
        model_name = int_column.replace("_int", "").replace("answer_", "")
        processed_df["batch_index"] = processed_df.index // batch_size

        batch_time_list = (
    processed_df.groupby("batch_index")["time_taken"].first().reset_index().to_dict(orient="records")
)


        total_time_taken = sum(item['time_taken'] for item in batch_time_list)
        
        avg_time_per_query =  total_time_taken/ len(df)
        evaluation_dict[model_name] = [avg_time_per_query, total_time_taken]


        df_combined = pd.concat([df_combined, processed_df], axis=1)

# print(df_combined)
df_combined = pd.concat([df_combined, ground_truth], axis=1)
df_combined['relationship'] = df_combined['relationship'].map({'yes': 1, 'no': 0})
df_combined['relationship'].fillna(1, inplace=True)
df_combined['pmid'] = df_combined['pmid'].astype(str)

df_combined = df_combined[~df_combined['pmid'].isin(dict_pmids)]
labels = df_combined["relationship"].to_numpy()
print(len(df_combined))
for model in df_combined.filter(regex='int').columns:
    model_pred = df_combined[model].to_numpy()
    print(len(model_pred))
    # Compute metrics
    tn, fp, fn, tp = confusion_matrix(labels, model_pred).ravel()
    specificity = tn / (tn + fp)
    precision = precision_score(labels, model_pred)
    recall = recall_score(labels, model_pred)
    npv = tn / (tn + fn)
    roc_auc = roc_auc_score(labels, model_pred)
    # Store results in evaluation dictionary
    model_name = model.replace("_int", "").replace("answer_", "")
    print(model_name)
    evaluation_dict[model_name].extend(  
        [np.round(roc_auc, 2), np.round(precision, 2), np.round(recall, 2), 
        np.round(specificity, 2), np.round(npv, 2)
    ])


# Convert evaluation dictionary to a DataFrame
evaluation_df = pd.DataFrame.from_dict(evaluation_dict, orient='index', 
                                       columns=["avg_time_per_query","total_time_taken", "ROC-AUC", "Precision", "Recall", "Specificity", "NPV"])
columns_order = ["ROC-AUC", "Precision", "Recall", "Specificity", "NPV", "avg_time_per_query", "total_time_taken"]
evaluation_df = evaluation_df[columns_order]




# Save the evaluation DataFrame
evaluation_df.to_csv(output_path, index=True)



# Define the batch size


