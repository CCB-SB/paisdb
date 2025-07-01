import os
import time
from os.path import abspath
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
    
torch.manual_seed(42) 

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,3"
os.environ['HF_HOME'] = '/local/paisdb/.cache/'

def open_source_query(model_id, pubmed_abstracts, batch_size):
    """Function to perfore inference using an open-source model from the transformers library that is specified via 'model_id'
    """    

    def create_query_batch(df, tokenizer):
        queries = []
        for pathogen, disease, title, abstract in zip(
            df["pathogen_term"].to_list(),
            df["disease_term"].to_list(),
            df["title_process"].to_list(),
            df["abstract_process"].to_list()
        ):            
            query_relationship = f"""I seek assistance with a systematic review focused on the direct relationship between pathogens and diseases, specifically {disease}. I’ll provide the title and abstract of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

1. The title and abstract provide sufficient evidence of a direct relationship between the disease ({disease}) and the pathogen ({pathogen}).
2. The title and abstract present data or findings supporting this association.

Exclusion criteria:
1. The title and abstract do not provide sufficient evidence of a direct relationship between the disease ({disease}) and the pathogen ({pathogen}).
2. The title and abstract lack primary data supporting the association.

Please provide the assessment in the following dictionary format:
{{"relationship": 1, "unrelated": 0}} if there is a relationship, or {{"relationship": 0, "unrelated": 1}} if the study should be excluded.

Note: only one value can be 1 at a time.

Title:{title}

Abstract: {abstract}

You are required to classify a journal article based solely on the given title and abstract. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the title and abstract.

Respond only in the dictionary format.

Answer:"""
            queries.append(query_relationship)
        
        queries_tokenized = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(1)
        
        return queries, queries_tokenized

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )
    # Device map = "auto" to parallelize the model over multiple devices using accelerate
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 offload_folder="offload", torch_dtype=torch.bfloat16)
                                                 #attn_implementation="flash_attention_2")
    model = model.eval()
    result_dict = {f"query_{model_id.split('/')[-1]}": [], f"answer_{model_id.split('/')[-1]}": [], "time_taken": []}
    batch_size = 50
    with torch.no_grad():
        for i in tqdm(range(0, len(pubmed_abstracts), batch_size)):
            print("LENGTH:" + str(len(pubmed_abstracts)))
            start_time = time.time()

            abstract_batch = pubmed_abstracts.iloc[i:i+batch_size]
            queries, inputs = create_query_batch(abstract_batch, tokenizer)    
            outputs = model.generate(**inputs, max_new_tokens=100)
            # outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.2, top_k=10, top_p=0.7, do_sample=True)
            # outputs = model.generate(**inputs, max_new_tokens=100, temperature=1, top_p=0.8, do_sample=True)
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            answers = [answer.split("Answer:")[-1].replace(" ", "").replace("\n", "") for answer in answers]

            end_time = time.time()
            time_elapsed = end_time - start_time
            num_items = len(answers)
            result_dict[f"query_{model_id.split('/')[-1]}"] += queries
            result_dict[f"answer_{model_id.split('/')[-1]}"] += answers
            result_dict["time_taken"] += [time_elapsed] * num_items  # Ensure consistent length


            print(result_dict[f"answer_{model_id.split('/')[-1]}"])

    result_df = pd.DataFrame.from_dict(result_dict)
    return result_df





