import os
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
import re
torch.manual_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["HF_HOME"] = "/local/paisdb/.cache/"

def open_source_query(model_id, pubmed_abstracts, batch_size):

    """
    Function to perform inference using an open-source model from the transformers library
    specified via 'model_id'.
    """
    batch_size = 10
    def create_query_batch( df, tokenizer):
        queries = []
        for pathogen, disease, title, abstract in zip(
            df["pathogen_term"].to_list(),
            df["disease_term"].to_list(),
            df["title_process"].to_list(),
            df["abstract_process"].to_list()
        ):
            query_relationship = f"""
**Task**: Classify a journal article based on whether it provides evidence of a direct relationship between {pathogen} and {disease}. Follow these steps:

1. **Analyze Title**:
   - Does the title explicitly mention both "{pathogen}" and "{disease}"?
   - Does it imply a relationship (e.g., "causes," "induces," "role in," "association with")?

2. **Analyze Abstract**:
   - Check if the abstract includes **any** of the following:
     - Direct claims of causation (e.g., "{pathogen} causes {disease}").
     - Investigation of {pathogen} with findings related to {disease} (e.g., "we studied {pathogen} and observed {disease} outcomes").
     - Study of {disease} with evidence implicating {pathogen} (e.g., "{pathogen} was identified in {disease} patients").
     - Brief mentions of an association (e.g., "{pathogen} has been linked to {disease}").
     - Data supporting the relationship (e.g., statistical results, experimental methods).

3. **Map to Criteria**:
   - **C1**: Direct causal claim in title/abstract.
   - **C2**: Abstract investigates {pathogen} and reports {disease} outcomes.
   - **C3**: Abstract investigates {disease} and identifies {pathogen} involvement.
   - **C4**: Abstract mentions association without focus.
   - **C5**: Abstract presents data (statistics, experiments, case studies).

4. **Exclusion Check**:
   - **E1**: No mention of both {pathogen} and {disease}, or explicitly states no relationship.

5. **Decision**:
   - If **ANY** of C1-C5 are met **AND** E1 is not triggered → **Include** (`{{"relationship": 1, "unrelated": 0}}`).
   - If **E1 applies** → **Exclude** (`{{"relationship": 0, "unrelated": 1}}`).

**Title**: {title}
**Abstract**: {abstract}

**Reasoning**:
- Title Analysis: [Mentions both? Implies relationship?]
- Abstract Findings: [Specific phrases/data related to criteria C1-C5]
- Criteria Met: [List which criteria apply]
- Exclusion Check: [Does E1 apply?]
- Final Decision: [Include/Exclude]

**Answer**: 
"""
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
        attn_implementation='flash_attention_2'
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
    print(model.device)

    # Prepare our result dictionary
    result_dict = {
        f"query_{model_id.split('/')[-1]}": [],
        f"answer_{model_id.split('/')[-1]}": [],
        "time_taken": []
    }

    # DO NOT overwrite the user-provided batch_size here

    with torch.no_grad():
        for i in tqdm(range(0, len(pubmed_abstracts), batch_size)):
            print(f"LENGTH: {len(pubmed_abstracts)}")

            start_time = time.time()

            abstract_batch = pubmed_abstracts.iloc[i:i+batch_size]
            queries, inputs = create_query_batch(abstract_batch, tokenizer)
            print(model.device)

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
            print(answers)
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

            print(result_dict[f"answer_{model_id.split('/')[-1]}"])
            

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
