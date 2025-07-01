import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"
os.environ['HF_HOME'] = '/local/paisdb/.cache/'

def open_source_query(model_id, pubmed_abstracts, batch_size=50):
    """Performs inference using a transformers model."""

    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with automatic device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        offload_folder="offload"
    )  # Do NOT move with .to(device), since device_map="auto" handles it
    model = model.eval()

    result_dict = {"query": [], "answer": [], "time_taken": []}

    def create_query_batch(df, tokenizer):
        queries = []
        for pathogen, disease, title, abstract in zip(
            df["pathogen_term"].to_list(),
            df["disease_term"].to_list(),
            df["title_process"].to_list(),
            df["abstract_process"].to_list()
        ):  
            query_relationship = f"""I seek assistance with a systematic review focused on the direct relationship between pathogens and diseases.

Inclusion criteria (Answer only {{"relationship": 1, "unrelated": 0}}) :
1. The title and abstract provide sufficient evidence of a direct relationship between the pathogen ({pathogen}) and the disease ({disease}).
2. The title and abstract present data or findings supporting this association.

Exclusion criteria (Answer only  {{"relationship": 0, "unrelated": 1}}):
1. The title and abstract do not provide sufficient evidence of a direct relationship between the pathogen ({pathogen}) and the disease ({disease}).
2. The title and abstract discuss drug resistance, treatment efficacy, laboratory experiments, or other contexts involving the pathogen without explicitly linking it as the cause of the disease.

Title: {title}

Abstract: {abstract}

You are required to classify a journal article based solely on the given title and abstract. Do not use any external knowledge.

Answer:
"""
            queries.append(query_relationship)
        
        # Ensure inputs are on the correct device
        queries_tokenized = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(device)
        return queries, queries_tokenized

    with torch.no_grad():
        for i in tqdm(range(0, len(pubmed_abstracts), batch_size)):
            start_time = time.time()
            abstract_batch = pubmed_abstracts.iloc[i:i+batch_size]
            queries, inputs = create_query_batch(abstract_batch, tokenizer)    

            # Ensure inputs are on the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            answers = [answer.split("Answer:")[-1].strip() for answer in answers]

            result_dict["query"] += queries
            result_dict["answer"] += answers
            result_dict["time_taken"] += [time.time() - start_time] * len(answers)

    return pd.DataFrame.from_dict(result_dict)
