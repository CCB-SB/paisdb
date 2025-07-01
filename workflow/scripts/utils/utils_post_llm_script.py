import sys
from os.path import abspath, join
import pandas as pd
import os
import ast
import re

# Optional: if you need to import from a scripts directory
sys.path.insert(0, abspath("scripts"))

# Mapping from string labels to integers
str2int = {"unrelated": 0, "relationship": 1}

# Code for invalid outputs
INVALID_CODE = 2



def process_result(result_df):
    """
    Processes answer columns in a DataFrame, extracting and converting JSON-like strings into integer mappings.

    Parameters:
    - result_df: Input DataFrame containing columns with 'answer' in their names.

    Returns:
    - DataFrame with processed columns suffixed by "_int".
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
                #print(extracted_dict)
                return max(str2int.get(key, 0) for key, value in extracted_dict.items() if value == 1)
            except:

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


def main():
    input_path = "/local/paisdb/results/relation_extraction/tmp/results_Qwen_Qwen2.5-72B-Instruct.csv"
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    print(f"Reading data from {input_path}")
    df = pd.read_csv(input_path)

    print("Processing results...")
    processed_df = process_result(df)
    print(processed_df)
    processed_df.columns = processed_df.columns.str.lower()


    output_path = "/local/paisdb/results/relation_extraction/final/zero/qwen.csv"
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()
