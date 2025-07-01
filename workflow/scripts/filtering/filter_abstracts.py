##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

from utils.utils_my_logger import My_logger

import re
from ast import literal_eval

##################################################
# FUNCTIONS
##################################################

def get_longest(my_list):
    lengths = [len(i) for i in my_list]
    index = lengths.index(max(lengths))
    
    return my_list[index]


def disease_regex_search(key_word, search_field):
    regex = re.compile(key_word.lower())
    return regex.search(search_field)

def bacteria_regex_search(key_word, search_field):
    regex = re.compile(key_word.lower())
    return regex.search(search_field)

# compares the taxonomy of to Pathogens returns true if pathogen1 is superclass of pathogen2
def compare_taxo(taxonomy_list1, taxonomy_list2):
    for taxo_path_1, taxo_path_2 in zip(taxonomy_list1, taxonomy_list2):
        if taxo_path_1 != taxo_path_2:
            if taxo_path_1 is None:
                return True
            else:
                return False
    return False

##################################################
# MAIN
##################################################
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from nltk.tokenize import sent_tokenize

    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "filtering")
    my_logger = logger.get_logger()
    
    df = pd.read_csv(str(snakemake.input.csv), index_col=0)

    # filter reviews
    #df_to_remove = df.loc[df['publication_type'].str.contains("Review", na=False)]
    #df = df.drop(df_to_remove.index)
    
    df['publication_date_pubmed'] = pd.to_datetime(df['publication_date_pubmed']).dt.strftime('%Y/%m/%d')
    df.title = df.loc[:, 'title'].astype(str)
    df.abstract = df.loc[:, 'abstract'].astype(str)
    my_logger.info(f"# Relations: {len(df.index)}")

    # Lower-case
    df[['disease_process', 'pathogen_process', 'abstract_process', 'title_process']] = df.loc[:, ['disease', 'pathogen', 'abstract', 'title']].map(str.lower)
    # Remove dots from spp.
    df['abstract_process'] = df.loc[:, 'abstract_process'].str.replace("spp.", "spp")
    df['title_process'] = df.loc[:, 'title_process'].str.replace("spp.", "spp")

    # Group by Pathogen-Disease term
    dispa = df.value_counts(subset=["disease", "pathogen", "query_key"]).reset_index()
    dispa.rename(columns={"count": "hits"}, inplace=True)

    minimun = min(dispa.hits)
    maximum = max(dispa.hits)

    try:
        dispa['query_group'] = pd.cut(dispa.hits, bins=pd.IntervalIndex.from_breaks([minimun, minimun+1, maximum]))
        my_logger.info(f"# Pathogen-Disease groups {dispa.value_counts(subset='query_group', normalize=True)}")
    except Exception as e:
        dispa['query_group'] = ''
        my_logger.info(f"Pathogen-Disease = min: {minimun}; max: {maximum}")
    
    # Filter abstract by length (limited by model resources)
    df_filt = df[df["abstract"].str.len() < 4000]

    #
    ## Identify which keyword (disease and pathogen) is present in the abstract
    #

    # List of Synonyms which are needed in the Query to get correct articles, but mask actual pathogen
    banned_synonyms = ["mosquito"]

    # Process query to obtain disease and pathogens synonyms when used
    queries = np.where(
        df_filt['query'].str.contains('OR') == False,
            None,
            df_filt['query'].str.replace(" AND fha[FILT]", ""
                                        ).str.replace(
                                            f""" AND ("{snakemake.params.date_collection_range[0]}"[Date - Publication] : "{snakemake.params.date_collection_range[1]}"[Date - Publication])""",
                                            "").str.replace("[TIAB]", ""
                                                    ).str.replace("(", "").str.replace(")", ""))
    queries_split = [ query.split(" AND ") if query else [None, None] for query in queries ]
    keys = [(d.split(" OR ") , p.split(" OR ")) if d and p else (None, None) for d, p in queries_split]
    diseases = [[i.replace('"', "").lower() for i in d] if d else None for d, _ in keys]
    pathogens = [[i.replace('"', "").lower() if i not in banned_synonyms else None for i in p] if p else None for _, p in keys]
    
    # Stats
    dis_syns = [i for i in diseases if i]
    patho_syns = [i for i in pathogens if i]
    my_logger.info(f"# queries using disease synonyms: {len(dis_syns)}")
    my_logger.info(f"# queries using pathogen synonyms: {len(patho_syns)}")

    # Infer which synonyms are in the abstract
    key_diseases = [[d for d in disease if (disease_regex_search(d,abstract) or disease_regex_search(d,title))] if disease else None for disease, abstract, title in zip(diseases, 
                                                                                                                        list(df_filt['abstract_process']), 
                                                                                                                        list(df_filt.title_process))]
    key_pathogens = [[p for p in pathogen if (bacteria_regex_search(p, abstract) or bacteria_regex_search(p, title))] if pathogen else None for pathogen, abstract, title in zip(pathogens,
                                                                                                                       list(df_filt['abstract_process']), 
                                                                                                                       list(df_filt.title_process))]
    
    # Replace not matches by None
    key_diseases = [ get_longest(i) if (i and len(i) > 0) else None for i in key_diseases]
    key_pathogens = [ get_longest(i) if (i and len(i) > 0) else None for i in key_pathogens]

    # Stats
    dis_match = [i for i in key_diseases if i]
    patho_match = [i for i in key_pathogens if i]
    my_logger.info(f"# disease synonyms matched in TIAB: {len(dis_match)}")
    my_logger.info(f"# pathogen synonyms matched in TIAB: {len(patho_match)}")

    # Replace None by original term
    key_diseases = [i if i else term for i, term in zip(key_diseases, list(df_filt['disease_process']))]
    key_pathogens = [i if i else term for i, term in zip(key_pathogens, list(df_filt['pathogen_process']))]

    # Insert terms
    df_filt['disease_term'] = key_diseases
    df_filt['pathogen_term'] = key_pathogens

    # Order by pubmed id just to put together possible repreated abstracts
    try:
        df_filt['pmid'] = df_filt.loc[:, 'pmid'].astype('int64')
    except Exception as e:
        print([df_filt.iloc[index] for index, i in enumerate(df_filt['pmid']) if pd.isna(i)])
        print(df_filt['query'].iloc[6403])
        raise e

    # ensure correct index needed for filter below
    df_filt.reset_index(inplace=True)

    df_patho = pd.read_csv(snakemake.input.pathogen, sep=",")
    df_patho = df_patho.replace({np.nan:None})

    pathogen_taxonomy_map = {}

    for pathogen, superkingdom_id,phylum_id,class_id,order_id,family_id,genus_id,species_id in zip(
        list(df_patho['pathogen']),
        list(df_patho['superkingdom_id']),
        list(df_patho['phylum_id']),
        list(df_patho['class_id']),
        list(df_patho['order_id']),
        list(df_patho['family_id']),
        list(df_patho['genus_id']),
        list(df_patho['species_id'])):
        pathogen_taxonomy_map[pathogen] = [superkingdom_id,phylum_id,class_id,order_id,family_id,genus_id,species_id]

    # Filter Pathogens Hits possible at genus and species level, keep species and link genus
    duplicate_rows_pathogen = []
    parent_link_pathogen = []
    for pmid in df_filt['pmid'].unique():
        # get entries for each pubmed id
        df_pmid = df_filt.loc[df_filt['pmid'] == pmid]
        for i, disease, pathogen in zip(range(len(df_pmid)), list(df_pmid['disease']), list(df_pmid['pathogen'])):
            if pathogen in pathogen_taxonomy_map:
                taxonomy = pathogen_taxonomy_map[pathogen]
                for j, disease2, pathogen2 in zip(range(len(df_pmid)), list(df_pmid['disease']), list(df_pmid['pathogen'])):
                    if i != j and disease == disease2 and pathogen2 in pathogen_taxonomy_map:
                        taxonomy2 = pathogen_taxonomy_map[pathogen2]
                        if compare_taxo(taxonomy, taxonomy2):
                            duplicate_rows_pathogen .append(df_pmid.index[i])
                            parent_link_pathogen.append((df_pmid.index[j], df_pmid.index[i]))

    # save parent class information in child
    parent = [None] * len(df_filt)
    for child_index, parent_index in parent_link_pathogen:
        parent[child_index] = df_filt['pathogen'][parent_index] if parent[child_index] is None else parent[child_index] + "+-+" + df_filt['pathogen'][parent_index]
    df_filt['parent_pathogen'] = parent

    # optain disease information for subclasses of each disease
    df_diseases = pd.read_csv(snakemake.input.disease, sep=",")
    df_diseases.dropna(subset="subclasses", inplace=True)
    

    # Regex for safer literaleval evaluation
    p = re.compile("{'name': .+, 'id': .+}", re.IGNORECASE)

    # create disease subclass mapping
    disease_subclass_map = {}
    for disease, subclass in zip(list(df_diseases['disease']), list(df_diseases['subclasses'])):
        regex_find = p.search(subclass)
        if regex_find:
            class_dict = literal_eval(subclass)
            subclass_list = [subclass_name.lower() for subclass_name in class_dict['name']]
            disease_subclass_map[disease] = class_dict['name']
        else:
            my_logger.info(f"Invalid dict syntax for {disease} while evaluating {subclass}")
    
    # Filter Diseases for Subclasses Example Liver cancer and hepatocellular carcinoma, keep hepatocellular carcinoma
    duplicate_rows_disease = []
    parent_link_disease = []
    for pmid in df_filt['pmid'].unique():
        # get entries for each pubmed id
        df_pmid = df_filt.loc[df_filt['pmid'] == pmid]
        for i, disease, pathogen in zip(range(len(df_pmid)), list(df_pmid['disease']), list(df_pmid['pathogen'])):
            disease = disease.lower()
            if disease in disease_subclass_map:
                subclasses = disease_subclass_map[disease]
                for j, disease2, pathogen2 in zip(range(len(df_pmid)), list(df_pmid['disease']), list(df_pmid['pathogen'])):
                    if i != j and pathogen == pathogen2 and disease2.lower() in subclasses:
                        disease2 = disease2.lower()
                        duplicate_rows_disease.append(df_pmid.index[i])
                        parent_link_disease.append((df_pmid.index[j], df_pmid.index[i]))

    # save parent class information in child
    parent = [None] * len(df_filt)
    for child_index, parent_index in parent_link_disease:
        parent[child_index] = df_filt['disease'][parent_index] if parent[child_index] is None else parent[child_index] + "+-+" + (df_filt['disease'][parent_index])
    df_filt['parent_disease'] = parent
    # Apply filters
    # remove duplicates removal via the unique indexes of duplicate_rows
    df_filt.drop(index=list(set(duplicate_rows_pathogen)), inplace=True)
    my_logger.info(f"Removed {len(set(duplicate_rows_pathogen))} duplicate / less specific entries")

    # From articles which have both the Disease and a Supclass of the same disease remove Disease 
    df_filt.drop(index=list(set(duplicate_rows_disease) - set(duplicate_rows_pathogen)), inplace=True)
    my_logger.info(f"Removed {len(set(duplicate_rows_disease) - set(duplicate_rows_pathogen))} duplicate / less specific entries filtering Diseases")

    df_filt.sort_values(by='pmid', inplace=True)
    df_filt.reset_index(drop=True, inplace=True)

    df_filt.to_csv(snakemake.output.csv, index=False)
    