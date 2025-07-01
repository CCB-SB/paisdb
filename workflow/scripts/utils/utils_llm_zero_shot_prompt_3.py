# Imports

# Setting enviroment
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["HF_HOME"] = '../.cache/'

import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
import re
torch.manual_seed(42)

API_TOKEN = 'insert_token_here'



def open_source_query(model_id, pubmed_abstracts, batch_size):

    """
    Function to perform inference using an open-source model from the transformers library
    specified via 'model_id'.
    """
    batch_size = 50
    def create_query_batch( df, tokenizer):
        queries = []
        for pathogen, disease, title, abstract in zip(
            df["pathogen_term"].to_list(),
            df["disease_term"].to_list(),
            df["title_process"].to_list(),
            df["abstract_process"].to_list()
        ):
            query_relationship = f"""I seek assistance with a systematic review focused on the direct relationship between pathogens and diseases, specifically {disease}. I’ll provide the title and abstract of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

1. The title or abstract provides sufficient evidence of a direct relationship between the disease ({disease}) and the pathogen ({pathogen}).
2. The title or abstract investigates the Pathogen ({pathogen}) and reports evidence for the Disease ({disease}).
3. The title or abstract investigates the Disease ({disease}) and reports evidence for the Pathogen ({pathogen}).
4. The title or abstract states the association between the Pathogen ({pathogen}) and the Disease ({disease}), but does not focus on it.
5. The title and abstract present data or findings supporting this association.

Exclusion criteria:
1. The title and abstract do not provide sufficient evidence of a direct relationship between the disease ({disease}) and the pathogen ({pathogen}).

Please provide the assessment in the following dictionary format:
{{"relationship": 1, "unrelated": 0}} if there is a relationship, or {{"relationship": 0, "unrelated": 1}} if the study should be excluded.

Note: only one value can be 1 at a time.

Title: {title}

Abstract: {abstract}

You are required to classify a journal article based solely on the given title and abstract. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the title and abstract.

Respond only in the dictionary format with no explanation.

Answer:"""
            queries.append(query_relationship)

        # If you plan to rely on device_map="auto", 
        # you can keep these tokenized inputs on CPU – accelerate will handle distribution.
        queries_tokenized = tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            padding_side='left'

        )
        return queries, queries_tokenized

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
        model_id,  # <-- Must be passed
        device_map="auto",               # <-- automatically sharded across visible GPUs
        offload_folder="offload",
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        local_files_only=True
    )

#     # 1) Load the *top-level* config from your Gemma3 checkpoint
#     full_config = Gemma3Config.from_pretrained(model_id)

#     # 2) Grab just the text sub-config
#     text_config = full_config.text_config

#     model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     config=text_config,                   # crucial!
#     device_map="auto",
#     offload_folder="offload",
#     torch_dtype=torch.bfloat16,
#     # attn_implementation="flash_attention_2",
# )

    model.eval()
    #print(model.device)

    # Prepare our result dictionary
    result_dict = {
        f"query_{model_id.split('/')[-1]}": [],
        f"answer_{model_id.split('/')[-1]}": [],
        "time_taken": []
    }

    # DO NOT overwrite the user-provided batch_size here

    with torch.no_grad():
        print(f"LENGTH: {len(pubmed_abstracts)}")
        for i in tqdm(range(0, len(pubmed_abstracts), batch_size)):

            start_time = time.time()

            abstract_batch = pubmed_abstracts.iloc[i:i+batch_size]
            queries, inputs = create_query_batch(abstract_batch, tokenizer)
            #print(model.device)

            inputs = {k: v.to("cuda") for k, v in inputs.items()}


            # If you like, explicitly move inputs to the model's device:
            # inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # outputs = model.generate(**inputs, max_new_tokens=300, eos_token_id=tokenizer.eos_token_id)
            # DEFUALT
            # outputs = model.generate(**inputs, max_new_tokens=300)

            # Option 1: Fully Deterministic (Greedy Decoding)
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
             do_sample=False, 
             eos_token_id=tokenizer.eos_token_id
            )

            # # Option 2: Minimal Randomness
            # outputs = model.generate(
            #     **inputs,
            #     max_new_tokens=300,
            #     temperature=0.1,
            #     top_k=5,
            #     top_p=0.95, do_sample=True
            # )

            # Option 3: Moderately Conservative
            # outputs= model.generate(
            #     **inputs,
            #     max_new_tokens=300,
            #     temperature=0.3,
            #     top_k=10,
            #     top_p=0.9, do_sample=True
            # )

            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # Clean up the answers
            answers = [
                ans.split("Answer:")[-1].replace(" ", "").replace("\n", "")
                for ans in answers
            ]
            #print(answers)
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
            # answers = [answer.split("Answer:")[-1].strip() for answer in answers]        
            end_time = time.time()
            time_elapsed = end_time - start_time
            num_items = len(answers)

            result_dict[f"query_{model_id.split('/')[-1]}"].extend(queries)
            result_dict[f"answer_{model_id.split('/')[-1]}"].extend(cleaned_answers)
            result_dict["time_taken"].extend([time_elapsed] * num_items)

            #print(result_dict[f"answer_{model_id.split('/')[-1]}"])
            

    result_df = pd.DataFrame.from_dict(result_dict)


    try: 
        sanitized_model_id = model_id.replace("/", "_")
        output_dir = '/local/paisdb/results/relation_extraction/tmp'
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'results_{sanitized_model_id}.csv')
        result_df.to_csv(output_path, index=False)
    except:
        pass
    return result_df
