##################################################
# IMPORTS
##################################################
import os
import pandas as pd
#from utils.utils import get_api

# Setting Device to use for training, has to be done before importing torch or transformers related packages
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
os.environ['HF_HOME'] = '../.cache/'
os.environ['HUGGINGFACE_TOKEN'] = 'insert_token_here'



# Use the token in the headers to authenticate
from huggingface_hub import login
from transformers import logging
login(os.environ['HUGGINGFACE_TOKEN'])  # Log in with the token
logging.set_verbosity_error()

import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))


from utils.LLM_based_extraction_zero_shot import open_source_query



##################################################
# FUNCTIONS
##################################################


def llm_based_approach():

    # Load articles
    pubmed_abstracts = pd.read_csv(str(snakemake.input.csv))

    result = open_source_query(model_id="mistralai/Mistral-Small-Instruct-2409",
                                      pubmed_abstracts=pubmed_abstracts, batch_size=50)

    return result

