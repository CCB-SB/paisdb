import sys
from os.path import abspath
import pandas as pd
import os
import ast
import re
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score


str2int = {"unrelated": 0, "relationship": 1}

sys.path.insert(0, abspath("scripts"))

def process_result(result_df):
    """
    Processes answer columns in a DataFrame, extracting and converting JSON-like strings into integer mappings.

    Parameters:
    - result_df: Input DataFrame containing columns with 'answer' in their names.

    Returns:
    - DataFrame with processed columns suffixed by "_int".
    """
    # Select columns containing 'answer' in their names
    result_df_int = result_df.filter(regex='answer')
    result_df_int = result_df_int.add_suffix("_int")

    # Function to safely extract and process dictionary-like strings
    def extract_and_map(answer):
        if pd.isnull(answer):
            return 0  # Default for missing data
        
        # Regex pattern to extract dictionary-like content
        pattern = r"\{.*?\}"
        match = re.search(pattern, str(answer))
        if match:
            dict_str = match.group()
            dict_str = dict_str.replace("'", '"')  # Ensure valid JSON format
            try:
                # Parse the dictionary and map the values
                extracted_dict = ast.literal_eval(dict_str)
                return max(str2int.get(key, 0) for key, value in extracted_dict.items() if value == 1)
            except:
                return 0  # Default for invalid dictionary content
        return 0  # Default if no dictionary is found

    # Process each column
    for col in result_df_int.columns:
        original_col = col.replace("_int", "")
        if original_col in result_df:
            result_df_int[col] = result_df[original_col].apply(extract_and_map)
        else:
            result_df_int[col] = 0  # Default if column is missing

    # Replace NaN with 0 in the final DataFrame
    result_df_int = result_df_int.fillna(0)

    return result_df_int

