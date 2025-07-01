import os
import time
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
import re

torch.manual_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["HF_HOME"] = "/local/paisdb/.cache/"

# ----------------------------------------------------------------------
# 1) Load your example data (same as your old code)
# ----------------------------------------------------------------------
with open('abstracts_titles_dict_file.json', 'r') as f:
    few_shot_dict = json.load(f)

# Build a string with multiple examples from your JSON file.
few_shot_examples = f"""
**Example 1**  
Query: {few_shot_dict.get("query1")}  
Answer: {few_shot_dict.get("answer1")}  
---

**Example 2**  
Query: {few_shot_dict.get("query4")}  
Answer: {few_shot_dict.get("answer4")}  

---

**Example 3**  
Query: {few_shot_dict.get("query3")}  
Answer: {few_shot_dict.get("answer3")}  

---
"""
# **Example 4**  
# Query: {few_shot_dict.get("query2")}  
# Answer: {few_shot_dict.get("answer2")}  

# ---

# **Example 5**  
# Query: {few_shot_dict.get("query5")}  
# Answer: {few_shot_dict.get("answer5")}  
# ---

# **Example 6**  
# Query: {few_shot_dict.get("query6")}  
# Answer: {few_shot_dict.get("answer6")}  
# ---
# """

def open_source_query(model_id, pubmed_abstracts, batch_size=50):
    """
    Function to perform inference using an open-source model from
    the transformers library (model_id), with your new prompt plus few-shot examples
    placed *within* that prompt text.
    """
    batch_size = 5 #Nemo
    # batch_size = 30 #Phi    
    # ------------------------------------------------------------------
    # 2) We'll embed the examples inline within your new prompt text.
    # ------------------------------------------------------------------
    def create_query_batch(df, tokenizer):
        queries = []
        for pathogen, disease, title, abstract in zip(
            df["pathogen_term"].to_list(),
            df["disease_term"].to_list(),
            df["title_process"].to_list(),
            df["abstract_process"].to_list()
        ):
            # Combine your new prompt "as is" with the examples, inline:
            prompt = f"""
I seek assistance with a systematic review focused on the direct relationship between pathogens and diseases, specifically {disease}. I’ll provide the title and abstract of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

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

Below are example classifications:

{few_shot_examples}

Title: {title}

Abstract: {abstract}

You are required to classify a journal article based solely on the given title and abstract. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the title and abstract.

Respond only in the dictionary format with no explanation.

Answer:
"""
            queries.append(prompt)

        # Tokenize for the model
        tokenized = tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            padding_side='left'
        )
        return queries, tokenized

    # 3) Tokenizer + 4-bit quant config
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 4) Load the model with device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2'
    )
    model.eval()

    # Prepare a result dictionary
    model_name_short = model_id.split('/')[-1]
    result_dict = {
        f"query_{model_name_short}": [],
        f"answer_{model_name_short}": [],
        "time_taken": []
    }

    # 5) Inference loop
    with torch.no_grad():
        for i in tqdm(range(0, len(pubmed_abstracts), batch_size)):
            start_time = time.time()

            abstract_batch = pubmed_abstracts.iloc[i:i+batch_size]
            queries, inputs = create_query_batch(abstract_batch, tokenizer)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Extract the dictionary from the text
            cleaned_answers = []
            for ans in answers:
                # After "Answer:"
                extracted = ans.split("Answer:")[-1].strip()
                # Look for { ... } dictionary
                match = re.findall(r'\{.*?\}', extracted)
                if match:
                    try:
                        cleaned_answers.append(eval(match[-1]))
                    except:
                        # Fallback if there's a parse error
                        cleaned_answers.append(extracted)
                else:
                    cleaned_answers.append(extracted)

            end_time = time.time()
            time_elapsed = end_time - start_time
            num_items = len(cleaned_answers)

            # Store results
            result_dict[f"query_{model_name_short}"].extend(queries)
            result_dict[f"answer_{model_name_short}"].extend(cleaned_answers)
            result_dict["time_taken"].extend([time_elapsed] * num_items)

            # Optional: Print answers for debugging
            print("BATCH ANSWERS:\n", cleaned_answers)

    # Return final DataFrame
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
