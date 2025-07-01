##################################################
# IMPORTS
##################################################
import sys
import os

from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

from torch.cuda.amp import autocast

import re
import numpy as np
import xml.etree.ElementTree as ET
from ast import literal_eval

from utils.utils_my_logger import My_logger

# import nltk

##################################################
# FUNCTIONS
##################################################

#### Functions for structured Analysis

# extracts xml tables
def table_extraction(document):
    tables = []
    for passage in document.passages:
        if 'xml' in passage.infons:
                # tables.append(passage.text) # Table in plaintext format
                tables.append(passage.infons['xml'])
    return tables

# removes stylisitc structures for example bold, itallic fonds from xml tables
def table_clean_up(table, patterns):
    for pattern in patterns:
        table = re.sub(f"</{pattern}>", "",table)
        table = re.sub(f"<{pattern}.*?>","", table)
    table = re.sub("\\\\n", "", table)
    return table

# given an xml table find rows which have informations about the pathogen
def table_scanning(table, pathogen, disease):
    # Patterns of stylistic choices which mask information in xml
    patterns = ['italic', 'bold', 'cursive']

    # perform clean up given the patterns
    table = table_clean_up(table, patterns)

    # create xml-tree of the table
    root = ET.fromstring(table)
    
    table_data = ""
    table_of_interest = False

    # table head if it exists
    if root.find("thead") is not None:
        for head in root.find("thead"):
            for row in head.iter("tr"):
                for colum in row.iter("td"):
                    if colum.text is not None:
                        table_data += colum.text + " | "
                    else:
                        table_data += "---" + " | "
        table_data += "\n"
    # Table body
    for tablerow in root.iter('tr'):
        table_row_data = ""
        contains_interest = False
        for tabledata in tablerow.iter('td'):
            table_data_text = tabledata.text
            if table_data_text is not None:
                table_data_text = table_data_text.lower()
                if pathogen or disease in table_data_text:
                    contains_interest = True
                table_row_data += table_data_text + " | "
        if contains_interest:
            table_of_interest = True
            table_data += table_row_data + "\n"

    return (1,table_data) if table_of_interest else (0,"")



#### Functions for unstructured Analysis

# returns the text content of passages which section_type is section name
def retrieve_section(document, section_name):
    text_collection = ""
    for passage in document.passages:
        if passage.infons['section_type'] == section_name:
            text_collection = text_collection + passage.text + '\n'
    return text_collection

# returns the text of passages which section_type is in the section_list
def retrieve_multiple_sections(collection, section_name_list):
    text_collection = ""
    for section_name in section_name_list:
        text_collection = text_collection + retrieve_section(collection, section_name)
    return text_collection

# decorator for regex substitution pattern = "<Word to Replace>" 
def replacePlaceholders(text, pathogen, disease, abstract, title):
    # check if input title is None and replace with empty string 
    if title is None:
        title = ""
        
    text = re.sub("<pathogen>", pathogen, text)
    text = re.sub("<disease>", disease, text)
    text = re.sub("<text>", re.escape(abstract), text)
    text = re.sub("<title>", re.escape(title), text)
    return text

def create_query_batch(df, tokenizer, messages, full_articles):

    instruction_list = []
    for pmc, pathogen, disease, abstract, title in zip(
            list(df['pmc']), 
            list(df['pathogen_term']), 
            list(df['disease_term']), 
            list(df['abstract_process']),
            list(df['title_process'])
        ):
        
        # Abstract is the baseline
        text = abstract

        # Build Questions using the message json file
        for i in range(len(messages['Messages'])):
            current_message = messages['Messages'][i]

            # Text data processing
            if pmc and pmc in full_articles: # Case where PMC was lost, but PubMed still has PMC
                document = full_articles[pmc]
                section_list = current_message['section'].split(',')
                text += '\n' + retrieve_multiple_sections(document, section_list)

            # create one instruction
            instruction = replacePlaceholders(current_message['user'], pathogen, disease, text, title)
            
            # Add to instructions
            instruction_list.append(instruction)

    instruction_tokenized = tokenizer(
        instruction_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side='left'
    )
    return instruction_list, instruction_tokenized

# cheks the LLM Answer only input text (Title and Text)
def checkAnswer(message, answer, text, logger):
    type = message['type']
    if type == 'int':
        # check if answer in txt
        if answer not in text:
            return None
        elif answer == 'None':
            return None
        elif str.isnumeric(answer):
            return int(answer) # works because there is no integer limit
        else:
            try:
                temp = answer.split('.')[0] # often Numbers are in float format
                temp = temp.replace(',', "") # remove commata seperations for large numbers
                answer = int(temp)
            except:
                logger.info(f"Conversion for type {type} failed for Answer {answer}, no conversion applied")
            return None
        
    elif type == 'generation':
        return answer.strip().replace("\"", "")
    
    elif type == 'string':
        if 'None' in answer:
            return None
        else:
            return answer.strip().replace("\"", "")
    
    elif type == 'bool':
        answer = answer.strip().replace("\"", "")
        if answer == '0' or answer == '1': # String numbers to int numbers
            return int(answer)
        if answer == 'No' or answer == 'Yes': # Convert Yes or No Questions
            return 1 if answer== 'Yes' else 0
        elif answer == 'True' or answer == 'False': # Convert True or False Questions
            return 1 if answer == 'True' else 0
        else:
            logger.info(f"Unknown conversion for {type} for Answer {answer}, no conversion applied")

    elif type == 'list':
        list_pattern = re.compile(r"\[.*?]")
        answer_str = list_pattern.search(answer)
        if answer_str:
            return answer_str.group()
        else:
            logger.info(f"Not in specified list format [.*] for Answer {answer}, no conversion applied")
            return []

    elif type == 'values':
        values = message['values']
        if answer in values:
            return answer
        else:
            return 'ND'
        

    else:
        logger.info(f"Unknown conversion type {type} for Answer {answer}, no conversion applied")
        return None


def llm_inference(model, inputs):
    start_time = time.time()
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generation without creativity
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode to optain human readable answers
    answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Clean up answers
    answers = [
        ans.split("Answer:")[-1].replace("\n", "")
        for ans in answers
    ]

    my_logger.info(f"Created {len(answers)} Answers from {len(outputs)} Outputs from {len(inputs.items())} inputs")

    return answers

def refine_answer(df, start_index, answers, messages, queries, my_logger):
    # add information into the csv
    answer_index = 0 # pos in the answer list

    # regex to ensure save literaleval evaluation
    p = re.compile(r"{.+?:.+?}", re.IGNORECASE)

    # index of the row in the dataframe from i to i + batchsize
    row_index = start_index

    while answer_index < len(answers):
        for message in messages['Messages']:
            # get the answer and remove new lines
            answer = answers[answer_index]
            # find the format {Name: Answer} and extract the answer
            answer_lst = p.findall(answer)
            if len(answer_lst) > 0:
                answer_value = answer_lst[-1].strip()[1:-1].split(":")[1].strip()
                answer_value = checkAnswer(message, answer_value, queries[answer_index], my_logger)
            else:
                my_logger.info(f"Not available : LLM Answer {answer}")
                answer_value = None

            column_index = df.columns.get_loc(message['name'])
            df.iat[row_index, column_index] = answer_value
            answer_index += 1
        row_index += 1

##################################################
# MAIN
##################################################
if __name__ == '__main__':
    import pandas as pd
    from bioc import biocxml
    from tqdm import tqdm

    import json
    import time

    from math import floor

    # Setting Device to use for training, has to be done before importing torch or transformers related packages
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
    os.environ['HF_HOME'] = '/local/paisdb_pipeline/.cache/'
    os.environ['HUGGINGFACE_TOKEN'] = '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'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Huggingface login
    from huggingface_hub import login
    login(os.environ['HUGGINGFACE_TOKEN'])  # Log in with the token

    # imports which might need os variables
    import torch
    torch.set_num_threads(2)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.pipelines.pt_utils import KeyDataset

    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "text mining")
    my_logger = logger.get_logger()

    # read message file
    with open("scripts/postprocessing/message.json") as json_file:
        my_logger.info("Loading Messages")
        messages = json.load(json_file)
        my_logger.info(f"Number of Messages: {len(messages['Messages'])} original")
        messages['Messages'] = [messages['Messages'][snakemake.params.message]]
        my_logger.info(f"Number of Messages: {len(messages['Messages'])} selected")
        my_logger.info(f"seleted Message: {messages['Messages'][0]['name']}")
        json_file.close()

    # create dictionary for full articles
    full_articles = {}
    with biocxml.iterparse(open(str(snakemake.input.xml), 'rb')) as reader:
        for document in reader:
            document_pmc = document.id if "PMC" in document.id else "PMC" + document.id
            full_articles[document_pmc] = document 

    # Model preparation
    model_id = "mistralai/Mistral-Small-Instruct-2409"
    # load tokanizer and token for padding
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Bits and Bytes Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # loading Model into Memory
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        local_files_only=True,
    )

    print(model.device)

    # read input file
    df = pd.read_csv(str(snakemake.input.csv))
    df = df.replace({np.nan:None})

    # Adding the fields for the dynamic Message file
    for message in messages['Messages']:
        df[message['name']] = [None] * len(df)

    # split df into pmc and non pmc entries
    df_no_pmc = df[df['pmc'].isna()]
    df_pmc = df[df['pmc'].notna()]
    
    # Do batchwise inference remember for each batch = size(Messages) * size(batch) can get large very fast
    with torch.no_grad():
        # for non pmc articles 50 total queries fit into the model
        batch_size = 50 // len(messages['Messages'])
        # first do non pmc fast

        print("Non PMC Batches")
        for i in tqdm(range(0, len(df_no_pmc), batch_size)):
            start_time = time.time()
            current_batch = df_no_pmc.iloc[i:i+batch_size]
            queries, inputs = create_query_batch(current_batch, tokenizer, messages, full_articles)
            answers = llm_inference(model, inputs)
            
            # extract answers from llm result and insert into df
            #refine_answer(df_no_pmc, i, answers, messages, queries, my_logger)

            end_time = time.time()
            time_elapsed = end_time - start_time
            my_logger.info(f"Finished Non PMC Batch {i}-{i+batch_size-1} in {time_elapsed}") 
        
        # second slow pmc
        batch_size = 10
        print("PMC Batches")
        for i in tqdm(range(0, len(df_pmc), batch_size)):
            start_time = time.time()
            current_batch = df_pmc.iloc[i:i+batch_size]
            queries, inputs = create_query_batch(current_batch, tokenizer, messages, full_articles)
            oom = False
            try:
                answers = llm_inference(model, inputs)
                
            except RuntimeError:
                my_logger.info(f"Batch {i}:{i+batch_size} to large, reduced batchsize to 1 for current batch")
                oom = True
                
            if oom:
                # repeat with lower batch size 
                answers = []
                queries = []
                
                for index in range(0,len(current_batch)):
                    single_row = current_batch.iloc[index : index + 1]
                    querie, inputs = create_query_batch(single_row, tokenizer, messages, full_articles)
                    answers.extend(llm_inference(model, inputs))

                    queries.append(querie)
                oom = False
            
            # extract answers from llm result and insert into df
            #refine_answer(df_pmc, i, answers, messages, queries, my_logger)

            end_time = time.time()
            time_elapsed = end_time - start_time
            my_logger.info(f"Finished PMC Batch {i}-{i+batch_size-1} in {time_elapsed}")  
    # merge back together
    df = pd.concat([df_no_pmc, df_pmc])
    
    

    '''
    # put into seperate file table evaluation
    for pmid, pmc, pathogen, disease in zip(list(df['pmid']),list(df['pmc']), list(df['pathogen_term']), list(df['disease_term'])):
        # full article table score
        if pmc in full_articles:
            document = full_articles[pmc] 
            table_score = 0
            tables = table_extraction(document)
            for table_index, table in enumerate(tables):
                score,finding = table_scanning(table, pathogen, disease)
                table_score += score
            result['table_score'].append(table_score)
        else:
            result['table_score'].append(None)
    '''

    # save results
    df.to_csv(snakemake.output.csv, sep=",", index=False)

