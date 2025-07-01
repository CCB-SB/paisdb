#!/usr/bin/env python
# coding: utf-8


##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

import re
from fuzzywuzzy import fuzz
from utils.utils_my_logger import My_logger

##################################################
# FUNCTIONS
##################################################

def process_term(term):
    # Change double to single quotes
    term = term.replace('"', "'")

    # Replace single quotes by nothing
    term = term.replace("'s", "").replace("'", "")

    return term

'''
Rules for Pathogen Disease Similarity
    A pathogen and Disease tuple are similar if either,
    1. the pathogen or Disease is fully in the other, but larger then 3 letters (avoid probabilistic similarity)
    2. They are similar in their wording upto small changes in the suffix or prefix (Edit Distance equal or greater than 0.85)
    3. The Similarity is not based on broad common terms
'''

def redundant_relation_pair(pathogen, disease, logger=None):
    # Rule 3
    pattern = " disease| fever| and| acute| syndrome| associated| virus| infection| adolescent"
    pathogen = re.sub(pattern, "", pathogen)
    disease = re.sub(pattern, "", disease)

    # Rule 1
    if (len(pathogen)>3 and len(disease)>3) and pathogen in disease or disease in pathogen:
        return True
    # Rule 2
    pathogen_parts = pathogen.split(" ")
    disease_parts = disease.split(" ")
    for p_part, d_part in product(pathogen_parts, disease_parts):
        if len(p_part) > len(d_part) and len(p_part) > 5:
            p_part = p_part[:-3]
        elif len(d_part) > len(p_part) and len(d_part) > 5:
            d_part = d_part[:-3]
        if fuzz.ratio(d_part, p_part) >= 85:
            return True

    return False


def redundant_relation(pathogen_names, disease_names, logger=None):
    for pathogen, disease in product(pathogen_names, disease_names):
        if redundant_relation_pair(pathogen.lower().strip(), disease.lower().strip()):
            return True

    return False

##################################################
# MAIN
##################################################

if __name__ == '__main__':
    from itertools import product
    import pandas as pd
    import numpy as np
    from os.path import join
    from utils.utils_concurrency import split_by_size
    import pickle

    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "ncbi_request")
    my_logger = logger.get_logger()

    #
    ## Create pubmed queries by combinining diseases and pathogens
    #

    # Causal relationships
    rela_df = pd.read_csv(snakemake.input.relations, na_values = None)
    rela_df = rela_df[rela_df['pathogen_type']=='cause']
    causal_relas = set(rela_df['pathogen']+':'+rela_df['disease'])
    my_logger.info(f"# causal relationship (pathogen-disease) = {len(causal_relas)}")

    # Pathogens
    df = pd.read_csv(snakemake.input.pathogens, na_values=None)
    df.replace(np.nan, None, inplace=True)
    # df = df[df['lowest_rank']!='superkindgom'] # Filter superkindgom level (too wide)
    df = df[df['pathogen'] != 'Sepsis'] # Filter sepsis microorganisms
    
    '''
    # filter less specific variants
    # sort via taxonomy with None values as smallest values
    df.sort_values(['superkingdom_id','phylum_id','class_id','order_id','family_id','genus_id','species_id'], ascending=True, inplace=True, na_position='first')
    keep_list = []
    # each row has to "proof" that it is the most specific taxonomy row by compairing it to the next row
    for i in range(0, df.shape[0]):
        if i + 1 == df.shape[0]:
            keep_list.append(True) # last row has no potential more specific taxonomy, hence it is by default keept 
            break
        else:
            df_compare = df.iloc[[i, i + 1]]
            df_compare_taxonomy = pd.DataFrame(df_compare, columns=['superkingdom_id', 'phylum_id', 'class_id', 'order_id', 'family_id', 'genus_id', 'species_id'])
            keep = False
            for colum in df_compare_taxonomy:
                # compare the values of each taxonomy level
                id1 = df_compare_taxonomy[colum].values[0]
                id2 = df_compare_taxonomy[colum].values[1]
                if pd.isna(id1):
                    if pd.isna(id2): # some times Nan / None in the lineage, but it continues. cannot do equal
                        continue
                    break
                elif id1 != id2:
                    keep = True
                    break
            keep_list.append(keep)
    # only keep most specific pathogen
    df = df.loc[keep_list]
    # resorting by pathogen name
    df.sort_values(by=['pathogen'], inplace=True)
    '''
    pathogens = [process_term(p) if p else p for p in list(df['pathogen'])]
    pathogens_syns = [process_term(syn) if syn else syn for syn in list(df['synonyms'])]
    my_logger.info(f"# pathogens = {len(pathogens)}")
    
    # Diseases
    df = pd.read_csv(snakemake.input.diseases, na_values=None)
    df.replace(np.nan, None, inplace=True)
    diseases = [process_term(d) if d else d for d in list(df['disease'])]
    diseases_syns = [process_term(syn) if syn else syn for syn in list(df['synonyms'])]
    my_logger.info(f"# diseases = {len(diseases)}")
    
    # Integrate synonyms of pathogens in the pubmed query
    pathogens_query_syns = {}
    pathogen_syn_map = {}
    for i, patho in enumerate(pathogens):
        query = f'"{patho}"[TIAB]'
        synonym = [patho]
        if pathogens_syns[i]:
            query = f"""({query} OR "{'"[TIAB] OR "'.join(pathogens_syns[i].split('|'))}"[TIAB])"""
            synonym += pathogens_syns[i].split('|')
        pathogens_query_syns[patho] = query
        pathogen_syn_map[patho] = synonym
    my_logger.info(f"Pathogens query integrating synonyms: {query}")

    # Integrate synonyms of diseases in the pubmed query
    diseases_query_syns = {}
    diseases_syn_map = {}
    for i, dis in enumerate(diseases):
        query = f'"{dis}"[TIAB]'
        synonym = [dis]
        if diseases_syns[i]:
            query = f"""({query} OR "{'"[TIAB] OR "'.join(diseases_syns[i].split('|'))}"[TIAB])"""
            synonym += diseases_syns[i].split('|')
        diseases_query_syns[dis] = query
        diseases_syn_map[dis] = synonym
    my_logger.info(f"Diseases query integrating synonyms: {query}")

    # Filter by Title/Abstract([TIAB]) AND Available Abstract (fha[FILT]) AND since 2004 AND Causal relationship
    queries_tup = [{'keywords': f'{dis},{patho}',
                    'disease': dis, 'pathogen': patho,
                    'query':f"""{diseases_query_syns[dis]} AND {pathogens_query_syns[patho]} AND fha[FILT] AND ("{snakemake.params.date_collection_range[0]}"[Date - Publication] : "{snakemake.params.date_collection_range[1]}"[Date - Publication])"""
                    } for dis, patho in product(diseases, pathogens) if f'{patho}:{dis}' not in causal_relas and not redundant_relation(pathogen_syn_map[patho] ,diseases_syn_map[dis], my_logger)]
    #
    ## Split queries in batches
    #
    ranges_batches = split_by_size(input=len(queries_tup), n=snakemake.params.batch_size)
    my_logger.info(f"Splitting {len(queries_tup)} PUBMED searches in {len(ranges_batches)} batches")
    batches = [(batch, queries_tup[i[0]:i[1]]) for batch, i in enumerate(ranges_batches)]

    for batch, query_tup_sub in batches:
        outfile = join(snakemake.output.DIR, f"batch_{batch}.pickle")
        pickle.dump(query_tup_sub, open(outfile, 'wb'))
