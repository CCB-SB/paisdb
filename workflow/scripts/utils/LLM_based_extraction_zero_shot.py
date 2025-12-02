# Imports

# Setting enviroment
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["HF_HOME"] = '../.cache/'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['HUGGINGFACE_TOKEN'] = 'insert_token_here'

import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
import re
torch.manual_seed(42)

from create_query_batches import batch_creator

API_TOKEN = 'insert_token_here'



def open_source_query(model_id, pubmed_abstracts, query_generator, batch_size=50):

    """
    Function to perform inference using an open-source model from the transformers library
    specified via 'model_id'.
    """


    # 1) Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 2) Prepare 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3) Load the model with device_map="auto" and pass in bnb_config
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",               # <-- automatically sharded across visible GPUs
        offload_folder="offload",
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        local_files_only=True # set this variable to False if the moddel needs to be downloaded, ensures otherwise not redownloading
    )

    model.eval()

    # Prepare our result dictionary
    result_dict = {
        f"query_{model_id.split('/')[-1]}": [],
        f"answer_{model_id.split('/')[-1]}": [],
        "time_taken": []
    }

    # select correct batch creator
    create_query_batch =batch_creator().get_query_batch(query_generator)

    with torch.no_grad():
        print(f"LENGTH: {len(pubmed_abstracts)}")
        for i in tqdm(range(0, len(pubmed_abstracts), batch_size)):

            start_time = time.time()

            abstract_batch = pubmed_abstracts.iloc[i:i+batch_size]
            queries, inputs = create_query_batch(abstract_batch, tokenizer)

            inputs = {k: v.to("cuda") for k, v in inputs.items()}


            # Fully Deterministic (Greedy Decoding)
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
             do_sample=False, 
             eos_token_id=tokenizer.eos_token_id
            )

            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # Clean up the answers
            answers = [
                ans.split("Answer:")[-1].replace(" ", "").replace("\n", "")
                for ans in answers
            ]

            cleaned_answers = []
            for ans in answers:
                extracted = ans.strip().split("Answer:")[-1]
                match = re.findall(r'\{.*?\}', extracted)  # Find all dictionary patterns
                if match:
                    try:
                        cleaned_answers.append(eval(match[-1]))  # Use the last match (final decision)
                    except:
                        cleaned_answers.append(ans)  # Default exclusion
                else:
                    cleaned_answers.append(ans)  # Default exclusion
            
            end_time = time.time()
            time_elapsed = end_time - start_time
            num_items = len(answers)

            result_dict[f"query_{model_id.split('/')[-1]}"].extend(queries)
            result_dict[f"answer_{model_id.split('/')[-1]}"].extend(cleaned_answers)
            result_dict["time_taken"].extend([time_elapsed] * num_items)
            

    result_df = pd.DataFrame.from_dict(result_dict)

    return result_df
