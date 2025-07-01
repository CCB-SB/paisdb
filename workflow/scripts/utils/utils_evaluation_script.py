import sys
from os.path import abspath
import pandas as pd
import os
import ast
import re
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
)
import numpy as np
import json
import sys
from os.path import abspath, join
import pandas as pd
import os
import ast
import re

sys.path.insert(0, abspath("scripts"))

# Mapping from string labels to integers
str2int = {"unrelated": 0, "relationship": 1}

# Code for invalid outputs
INVALID_CODE = 2

suffix = 'COT'
batch_size = 10 #zero
# batch_size = 100 #mini_zero
# batch_size = 50 #mini_few
# batch_size = 50 #mini_few
# batch_size = 50 #mini_few_6
# batch_size = 5 #few
# batch_size = 3 #few_6
# batch_size = 3 #few_6
batch_size = 10 #COT
# Input paths
llm_directory = f"/local/paisdb/results/relation_extraction/final/{suffix}/"
ground_truth_path = f"/local/paisdb/src/final_test_dataset_12_03_2025_results.csv"
output_path = f"/local/paisdb/results/relation_extraction/final/{suffix}/evaluation_2.csv"
results_output_path = f"/local/paisdb/results/relation_extraction/final/{suffix}/results_{suffix}.csv"



def process_result(result_df):
    """
    Processes answer columns in a DataFrame, extracting and converting JSON-like
    strings into integer mappings.

    Returns a DataFrame with processed columns suffixed by "_int".
    """
    result_df_int = result_df.filter(regex='answer')
    result_df_int = result_df_int.add_suffix("_int")

    def extract_and_map(answer):
        if pd.isnull(answer):
            return INVALID_CODE  # Default for missing data

        pattern = r"\{.*?\}"
        match = re.search(pattern, str(answer))
        if match:
            dict_str = match.group()
            dict_str = dict_str.replace("'", '"')  # Make JSON-like
            try:
                extracted_dict = ast.literal_eval(dict_str)

                print(extracted_dict)


                # Return 1 if "relationship":1 found, 0 if "unrelated":1 found, else INVALID
                return max(
                    str2int.get(key, 0)
                    for key, value in extracted_dict.items()
                    if value == 1
                )
            except:
                print("Error in extracting dict" + dict_str)
                return INVALID_CODE
        return INVALID_CODE

    for col in result_df_int.columns:
        original_col = col.replace("_int", "")
        if original_col in result_df:
            result_df_int[col] = result_df[original_col].apply(extract_and_map)
        else:
            result_df_int[col] = INVALID_CODE

    result_df_int = result_df_int.fillna(INVALID_CODE)
    final_df = pd.concat([result_df, result_df_int], axis=1)
    return final_df

# Load your data dictionary
with open('abstracts_titles_dict_file.json', 'r') as file:
    data_dict = json.load(file)

dict_pmids = set(value for key, value in data_dict.items() if "pmid" in key)
print(dict_pmids)

def get_column_with_int(df):
    """Return the first column name in df that contains 'int' in its name."""
    for col in df.columns:
        if 'int' in col:
            return col
    return None


print(output_path)

# Load the ground truth file
ground_truth = pd.read_csv(ground_truth_path)

# Ensure ground truth has a 'relationship' column
if "relationship" not in ground_truth.columns:
    raise ValueError("'relationships.csv' must contain a 'relationship' column.")

# Dictionary to store final metrics {model_name: [list_of_metrics]}
# We'll store 2 items for time stats + 7 items for metrics = 9 total
evaluation_dict = {}
df_combined = pd.DataFrame()

# Iterate over CSV files in the directory
for file in os.listdir(llm_directory):
    # Skip evaluation files
    if file.endswith("evaluation.csv") or file.endswith("evaluation_2.csv"):
        continue
    if file.endswith(".csv"):
        file_path = os.path.join(llm_directory, file)
        print(file_path)
        df = pd.read_csv(file_path)
        print(len(df))

        # Drop any pre-existing '_int' columns
        df = df.drop(df.filter(regex='int').columns, axis=1)

        # Process the result (returns original + *_int columns)
        processed_df = process_result(df)

        # Bring back your 'time_taken' column
        processed_df = pd.concat([processed_df, df['time_taken']], axis=1)

        # Identify the primary integer column
        int_column = get_column_with_int(processed_df)
        # E.g., "answer_X_int" => model_name = "X"
        model_name = int_column.replace("_int", "").replace("answer_", "")

        # Compute average time per query
        if "phi" in model_name and "few" in suffix:
            print("phiphiphi")
            batch_size = 30
        processed_df["batch_index"] = processed_df.index // batch_size
        batch_time_list = (
            processed_df.groupby("batch_index")["time_taken"]
            .first()
            .reset_index()
            .to_dict(orient="records")
        )
        total_time_taken = sum(item['time_taken'] for item in batch_time_list)
        avg_time_per_query = total_time_taken / len(df)

        # Initialize 2 items in the metrics list
        evaluation_dict[model_name] = [avg_time_per_query, total_time_taken]

        # Accumulate results into a single DataFrame for final evaluation
        df_combined = pd.concat([df_combined, processed_df], axis=1)
        int_columns = [col for col in df_combined.columns if col.endswith('_int')]

        # Create a new DataFrame with only those columns
        df_int = df_combined[int_columns].copy()
        print(df_int)

        # Add a suffix to each column name
          # Change this to whatever string you want to append
        df_int.columns = [col + suffix for col in df_int.columns]

        df_int.to_csv(results_output_path, index=True)

# Append ground truth columns
df_combined = pd.concat([df_combined, ground_truth], axis=1)

# Convert ground truth to 0/1
df_combined['relationship'] = df_combined['relationship'].map({'yes': 1, 'no': 0})
df_combined['relationship'].fillna(1, inplace=True)
df_combined['pmid'] = df_combined['pmid'].astype(str)

# Filter out PMIDs in your dict
df_combined = df_combined[~df_combined['pmid'].isin(dict_pmids)]
labels = df_combined["relationship"].to_numpy()
print("Final df_combined length:", len(df_combined))

# Evaluate each model column that ends with _int
for model_col in df_combined.filter(regex='int').columns:
    # Model predictions
    model_pred = df_combined[model_col].to_numpy()

    # Extract model name
    model_name = model_col.replace("_int", "").replace("answer_", "")

    # 1) Identify invalid predictions (== 2)
    invalid_mask = (model_pred == 2)
    num_invalid = invalid_mask.sum()
    frac_invalid = (num_invalid / len(model_pred)) * 100.0

    # 2) Filter out invalid predictions for binary metrics
    valid_mask = ~invalid_mask
    valid_labels = labels[valid_mask]
    valid_preds = model_pred[valid_mask]

    # If everything is invalid => store placeholders
    if len(valid_preds) == 0:
        print(f"[WARNING] All predictions invalid for model: {model_name}.")
        # We'll add 7 metrics, including MacroF1 as NaN
        evaluation_dict[model_name].extend([
            np.nan,    # ROC-AUC
            np.nan,    # Precision
            np.nan,    # Recall
            np.nan,    # Macro-F1
            np.nan,    # Specificity
            np.nan,    # NPV
            frac_invalid  # fraction invalid
        ])
        continue

    print(f"Model: {model_name}, valid predictions: {len(valid_preds)}")

    # 3) Compute metrics
    tn, fp, fn, tp = confusion_matrix(valid_labels, valid_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    precision = precision_score(valid_labels, valid_preds, zero_division=0)
    recall = recall_score(valid_labels, valid_preds, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else 0
    # If labels are all 0 or all 1 => can't compute AUC
    roc_auc = (
        roc_auc_score(valid_labels, valid_preds)
        if len(np.unique(valid_labels)) > 1
        else np.nan
    )
    # Compute macro-F1
    macro_f1 = f1_score(valid_labels, valid_preds, average="macro")

    # 4) Extend with 7 items => total 9
    # (2 items from above + 7 = 9 total per model)
    evaluation_dict[model_name].extend([
        round(roc_auc, 2) if roc_auc is not np.nan else np.nan,  # ROC-AUC
        round(precision, 2),                                     # Precision
        round(recall, 2),                                        # Recall
        round(macro_f1, 2),                                      # Macro-F1
        round(specificity, 2),                                   # Specificity
        round(npv, 2),                                           # NPV
        frac_invalid                                             # percentage
    ])

# Now build the final DataFrame with exactly 9 columns
evaluation_df = pd.DataFrame.from_dict(
    evaluation_dict,
    orient='index',
    columns=[
        "avg_time_per_query",  # 1
        "total_time_taken",    # 2
        "ROC-AUC",             # 3
        "Precision",           # 4
        "Recall",              # 5
        "MacroF1",             # 6
        "Specificity",         # 7
        "NPV",                 # 8
        "FracInvalid"          # 9
    ]
)

# Reorder columns if desired
columns_order = [
    "ROC-AUC", "Precision", "Recall", "MacroF1",
    "Specificity", "NPV", "FracInvalid",
    "avg_time_per_query", "total_time_taken"
]
evaluation_df = evaluation_df[columns_order]

# Save evaluation
evaluation_df.to_csv(output_path, index=True)
print("Evaluation complete. Results saved to:", output_path)
