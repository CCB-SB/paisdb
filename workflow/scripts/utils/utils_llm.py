def open_source_query(model_id, pubmed_abstracts, batch_size):
    """Function to perfore inference using an open-source model from the transformers library that is specified via 'model_id'
    """    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    import pandas as pd
    import torch
    from tqdm import tqdm
    
    def create_query_batch(df, tokenizer):
        queries = []
        abstract_series, disease_series, pathogen_series = df["abstract"], df["query_key"].str.split(",").str[0], df["query_key"].str.split(",").str[1]
        for abstract, disease, pathogen in zip(abstract_series.to_list(), disease_series.to_list(), pathogen_series.to_list()):
            query_relationship = f"""You will be provided with the abstract of a scientific article.
                Judge from the given abstract, if there is sufficient evidence that:
                   1) The disease {disease} and pathogen {pathogen} are associated(answer only 'relationship');
                or 2) The disease {disease} and the pathogen {pathogen} are not associated or there is no sufficient evidence for a relationship (answer only 'unrelated').
                Again, your answer should only be  'relationship' or 'unrelated', depending on your decision, and nothing else.
                Abstract: {abstract}
                Answer:
                """

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
                                                 device_map="auto",  
                                                 quantization_config=bnb_config) 
                                                 #attn_implementation="flash_attention_2")
    model = model.eval()
    result_dict = {f"query_{model_id.split('/')[-1]}": [], f"answer_{model_id.split('/')[-1]}": []}
    #TODO: turn pubmed abstracts into dataloader and increase batch size
    # ror i in range(0,len(df), batch_size)
    # batch df = df.iloc[i:i+batch_size]
    batch_size = batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(pubmed_abstracts), batch_size)):

            abstract_batch = pubmed_abstracts.iloc[i:i+batch_size]
            queries, inputs = create_query_batch(abstract_batch, tokenizer)    
       # """
       # #TODO: include causality query
       # #query_causality = f"""
        #You will be provided with the abstract of a scientific article.
        #Judge from the given abstract, if there is sufficient evidence that: 1) The disease {disease} is caused by the pathogen {pathogen}
        #(answer only 'pathogen causes disease'); or 2) The pathogen {pathogen} is caused by the disease {disease}  (answer only 'disease causes pathogen'). If there is no sufficient evidence to judge about causality just from the abstract, answer only 'inconclusive'. Again, your answer should only be  'relationship' or 'unrelated', depending on your decision, and nothing else.
        #"""

        #    inputs = tokenizer(query_relationship, return_tensors="pt").to(1)
            outputs = model.generate(**inputs, max_new_tokens=20)
            
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            answers = [answer.split("Answer:")[-1].replace(" ", "").replace("\n", "") for answer in answers]
            result_dict[f"query_{model_id.split('/')[-1]}"] += queries
            result_dict[f"answer_{model_id.split('/')[-1]}"] += answers

            print(result_dict[f"answer_{model_id.split('/')[-1]}"])

    result_df = pd.DataFrame.from_dict(result_dict)
    return result_df

def process_result(result_df):
    """ Function to turn the prediction strings into numerical labels,
    predictions that do not provide the right answer format always predict zero,
    check for GPT-3.5 if thats the case

    NOTE: was previously in the wrong order,
    for new predictions please change the corresponding cell in notebook/prediction_eval.ipynb
    """

    import pandas as pd

    str2int = {"unrelated": 0, "relationship": 1}
    result_df_int  = result_df.filter(regex='answer')
    result_df_int = result_df_int.add_suffix("_int")
    
    result_df_int = pd.concat([result_df_int[col].str.lower().map(str2int) for col in result_df_int.columns], axis=1)
    result_df_int = result_df_int.fillna(0)
    return result_df_int
