##################################################
# IMPORTS
##################################################
import os
from os.path  import abspath
# Setting Device to use for training, has to be done before importing torch or transformers related packages
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

import sys
sys.path.insert(0, abspath("scripts"))

from utils.utils_my_logger import My_logger
from utils.utils_llm import open_source_query, process_result

##################################################
# FUNCTIONS
##################################################

def query(client, PROMPT, model, MaxToken=50): 
    """Helper function that actually performs the query
    """
    response = client.chat.completions.create( 
        model=model,  
        messages=PROMPT, 
        max_tokens=MaxToken, 
    ) 
    answer = response.choices[0].message.content
    
    return answer

def query_df(df, df_id, model, my_logger=None):
    """Query the OpenAI API with a chunk of the data
    """
    import pandas as pd
    import numpy as np
    import multiprocessing

    my_logger.info(f"""Process {multiprocessing.current_process().name} started working on task {df_id},
                   processing {len(df)} queries""")

    result_dict = {f"query_{model}": [], f"answer_{model}": []}

    for i, row in df.iterrows():
        abstract = row["abstract"]
        disease, pathogen = row["query_key"].split(",")[0], row["query_key"].split(",")[1]

        query_full = f"""Abstract: {abstract}
                    Answer:"""
    
        message_full = [
            {"role": "system", "content": 
            
            f"""You will be provided with the abstract of a scientific article. 
        Judge from the given abstract, if there is sufficient evidence that: 1) The disease {disease} and pathogen {pathogen} are associated
        (answer only 'relationship'); or 2) The disease {disease} and the pathogen {pathogen} are not associated (answer only 'unrelated'). If there is not sufficient evidence in either direction answer only 'inconclusive'. Again, your answer should only be  'relationship', 'unrelated' or 'inconclusive', depending on your decision."""
            
            },

            {"role": "user", "content": query_full}
        ]

        try: 
            answer_full = query(message_full, model, MaxToken=2000)
            result_dict[f"query_{model}"].append(query_full)
            result_dict[f"answer_{model}"].append(answer_full)
           
        except:
            my_logger.error(f"Query {query_full} lead to an error")
            result_dict[f"query_{model}"].append(np.nan)
            result_dict[f"answer_{model}"].append(np.nan)
        
        if i % 500 == 0:
            my_logger.info(f"{multiprocessing.current_process().name} finished processing {i} queries")
    
    my_logger.info(f'Process {multiprocessing.current_process().name} ended working on task {df_id}')
    
    # Save results
    result_df = pd.DataFrame.from_dict(result_dict)
    
    return result_df


def openai_query(model, pubmed_abstracts, n_processes):
    """Function to perform queries to the OpenAI API in parallelized fashion
    """
    import multiprocessing
    import pandas as pd

    chunk_size = int(len(pubmed_abstracts)/n_processes)
    print(chunk_size)
    chunks = [(pubmed_abstracts.iloc[pubmed_abstracts.index[chunk_border:chunk_border + chunk_size]].reset_index(drop=True), chunk_id, model) 
              if chunk_border+chunk_size < len(pubmed_abstracts) 
              else (pubmed_abstracts.iloc[pubmed_abstracts.index[chunk_border:len(pubmed_abstracts)]].reset_index(drop=True), chunk_id, model) for chunk_id, chunk_border in enumerate(range(0, len(pubmed_abstracts), chunk_size))]
    
    pool = multiprocessing.Pool(processes=n_processes)
    result = pool.starmap(query_df, chunks)
    final_result = pd.concat(result).reset_index(drop=True)
    
    return final_result
 
##################################################
# MAIN
##################################################
if __name__ == '__main__':

    from utils.utils import get_api
    from openai import OpenAI
    import pandas as pd

    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "llm-benchmarking")
    my_logger = logger.get_logger()

    # set openAI API KEY
    #openai_token = get_api(
    #    api_file = snakemake.params.api_file, header = snakemake.params.api_key)
    openai_token = "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"
    client = OpenAI(api_key=openai_token)
    
    # Load articles
    pubmed_abstracts = pd.read_csv(snakemake.input.csv)
    pubmed_abstracts = pubmed_abstracts.iloc[:snakemake.params.n_queries]
    print(len(pubmed_abstracts))
    pubmed_abstracts = pubmed_abstracts[pubmed_abstracts["abstract"].str.len() < 4000]
    print(len(pubmed_abstracts))

    # run api queries
    result_gpt3_5 = openai_query(model="gpt-3.5-turbo", pubmed_abstracts=pubmed_abstracts,
                                 n_processes=snakemake.threads)
    result_gpt4 = openai_query(model="gpt-4", pubmed_abstracts=pubmed_abstracts,
                               n_processes=snakemake.threads)

    # run open-source queries
    import time
    
    start_llama2 = time.time()
    result_llama2 = open_source_query(model_id="upstage/Llama-2-70b-instruct-v2",
                                  pubmed_abstracts=pubmed_abstracts,
                                  batch_size=64)
    end_llama2 = time.time() 
    llama2_time = end_llama2 - start_llama2
    llama2_time_per_abstract = llama2_time / len(pubmed_abstracts)

    start_mixtral = time.time()
    result_mixtral = open_source_query(model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                                       pubmed_abstracts=pubmed_abstracts, 
                                       batch_size=128)
    end_mixtral = time.time()
    mixtral_time = end_mixtral - start_mixtral
    mixtral_time_per_abstract = mixtral_time / len(pubmed_abstracts)

    time_df = pd.DataFrame.from_dict({"Llama seconds/abstract": [llama2_time_per_abstract], "Mixtral seconds/abstract": [mixtral_time_per_abstract]})
    time_df.to_csv(snakemake.output.time_csv)
    # Merge results
    final_result = pd.concat((pubmed_abstracts, 
                              result_gpt3_5, 
                              result_gpt4, 
                              result_llama2, 
                              result_mixtral), axis=1)
    final_result = pd.concat((final_result, process_result(result_df=final_result)), axis=1)
  
    final_result.to_csv(snakemake.output.result_csv, index=False)
   
