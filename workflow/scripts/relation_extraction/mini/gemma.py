##################################################
# IMPORTS
##################################################
import os
import pandas as pd
# Setting Device to use for training, has to be done before importing torch or transformers related packages
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
os.environ['HF_HOME'] = '/local/paisdb/.cache/'
os.environ['HUGGINGFACE_TOKEN'] = 'insert_token_here'

# Use the token in the headers to authenticate
from huggingface_hub import login
login(os.environ['HUGGINGFACE_TOKEN'])  # Log in with the token

import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))
from utils.utils_post_llm import process_result    

mode = snakemake.params.mode



if mode == "few-shot":
    from utils.utils_llm_few_shot import open_source_query
    print("Running in few-shot-prompt...")
elif mode == "zero-shot-prompt-1":
    from utils.utils_llm_zero_shot_prompt_1_vllm import open_source_query
    print("Running in zero-shot-prompt-1...")
elif mode == "zero-shot-prompt-2":
    from utils.utils_llm_zero_shot_prompt_2 import open_source_query
    print("Running in zero-shot-prompt-2...")
elif mode == "zero-shot-prompt-3":
    from utils.utils_llm_zero_shot_prompt_3 import open_source_query
    print("Running in zero-shot-prompt-3...")
else: 
    from utils.utils_llm_COT_prompt import open_source_query
    print("Running in COT mode...")




##################################################
# FUNCTIONS
##################################################



##################################################
# MAIN
##################################################
if __name__ == '__main__':


    # Load articles
    pubmed_abstracts = pd.read_csv(snakemake.input.csv)
    pubmed_abstracts = pubmed_abstracts.iloc[:snakemake.params.n_queries]

    # run open-source queries
    result = open_source_query(model_id="google/gemma-2-9b-it",
                                      pubmed_abstracts=pubmed_abstracts, batch_size=2)

    # Merge results
    final_result = pd.concat((result, process_result(result_df=result)), axis=1)
    final_result.to_csv(snakemake.output.csv, index=False)
