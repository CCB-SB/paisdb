##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

import re
from fuzzywuzzy import fuzz

from utilsmeta import utils, parse_ete4

##################################################
# FUNCTIONS
##################################################

def get_longest(my_list):
    lengths = [len(i) for i in my_list]
    index = lengths.index(max(lengths))
    
    return my_list[index]


def disease_regex_search(key_word, search_field):
    # Expandable rule set
    rule_set = []
    rule_set.append(f"{key_word.replace(" ", "( |-)")}") # keyword itself
    rule_set.append(f"{(key_word[0].upper() + key_word[1:]).replace(" ", "( |-)")}") # first letter capitalised

    # concatinate rules
    rule_string = ""
    for rule in rule_set:
        rule_string += rule + "|"

    # Remove last pipe
    final_rule_string = rule_string[:-1]

    # generate Regex and make search
    regex = re.compile(final_rule_string)
    return regex.search(search_field)

def bacteria_regex_search(key_word, search_field):
    # Expandable rule set
    rule_set = []
    rule_set.append(f"{key_word.replace(" ", "( |-|\\w{0,15})")}") # keyword itself
    rule_set.append(f"{(key_word[0].upper() + key_word[1:]).replace(" ", "( |-)")}") # first letter capitalised

    # concatinate rules
    rule_string = ""
    for rule in rule_set:
        rule_string += rule + "|"

    # Remove last pipe
    final_rule_string = rule_string[:-1]

    # generate Regex and make search
    regex = re.compile(final_rule_string)
    return regex.search(search_field)

# function for removing duplicates from a Batch of abstracts
def remove_duplicates(df):
    # filter duplicates
    ETE4 = parse_ete4.NCBITaxa_mod()
   
    duplicate_rows = []
    for pmid in df['pmid'].unique():
        # get entries for each pubmed id
        df_pmid = df.loc[df['pmid'] == pmid]
        # create a list of all lineages of named bacteria from a pubmed entry
        pathogen_lineages_list = []
        for pathogen in df_pmid['pathogen']:
            pathogen_id = ETE4.get_name_translator([pathogen]).get(pathogen)[0]
            pathogen_lineage = ETE4.get_lineage(pathogen_id)
            pathogen_lineages_list.append(pathogen_lineage)

        # iterate over each entry in the list and check if it is contained in another entry --> duplicate with less information
        # its index is stored in the duplicate_rows
        for i in range(len(pathogen_lineages_list)):
            for j in range(len(pathogen_lineages_list)):
                if i != j and len(pathogen_lineages_list[i]) <= len(pathogen_lineages_list[j]):
                    boolean_list = [pathogen_lineages_list[i][l] == pathogen_lineages_list[j][l] for l in range(len(pathogen_lineages_list[i]))]
                    if all(boolean_list):
                        duplicate_rows.append(df_pmid.index[i])

    # remove duplicates removal via the unique indexes of duplicate_rows
    df.drop(index=list(set(duplicate_rows)), inplace=True)
    return list(set(duplicate_rows))

''' 
    function for removing duplicates from the reference
    @param reference The reference dataframe
    @param duplicate_list The list of duplicate entries obtained through remove_duplicates()
    @param ref_batch The Integer Representing the Batch of the duplicate_list

'''
def remove_duplicates_in_reference(reference, duplicate_list, ref_batch):
    index_list = []
    for index, abstract, batch in zip(reference.index, reference['Extraction Abstract'], reference['Batch']): 
        if abstract in duplicate_list and batch == ref_batch:
            index_list.append(index)
    reference.drop(index_list, inplace=True)

##################################################
# MAIN
##################################################
if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # reading test data
    df_reverence = pd.read_csv("../workflow/tests/filtering/synonyms/resources/reference.csv", sep=";", keep_default_na=False)
    df_batch_5 = pd.read_csv("../workflow/tests/filtering/synonyms/resources/batch_5.csv", index_col=0)
    df_batch_8 = pd.read_csv("../workflow/tests/filtering/synonyms/resources/batch_8.csv", index_col=0)
    
    # remove duplicates
    duplicates_5 = remove_duplicates(df_batch_5)
    duplicates_8 = remove_duplicates(df_batch_8)

    # Join the batches
    df_test = pd.concat([df_batch_5, df_batch_8])

    # clean up
    df_reverence['Used Name'] = [name if name != '' else None for name in df_reverence['Used Name']]
    df_reverence.drop(df_reverence[df_reverence['Remark']=="abstract to long"].index, inplace=True)

    # remove duplicate entries in the reference 
    remove_duplicates_in_reference(df_reverence, duplicates_5, 5)
    remove_duplicates_in_reference(df_reverence, duplicates_8, 8)

    df_test['publication_date_pubmed'] = pd.to_datetime(df_test['publication_date_pubmed']).dt.strftime('%Y/%m/%d')
    df_test.title = df_test.loc[:, 'title'].astype(str)
    df_test.abstract = df_test.loc[:, 'abstract'].astype(str)

    # Lower-case
    df_test[['disease_process', 'pathogen_process', 'abstract_process', 'title_process']] = df_test.loc[:, ['disease', 'pathogen', 'abstract', 'title']].map(str.lower)
    # Remove dots from spp.
    df_test['abstract_process'] = df_test.loc[:, 'abstract_process'].str.replace("spp.", "spp")
    df_test['title_process'] = df_test.loc[:, 'title_process'].str.replace("spp.", "spp")
    
    # Filter abstract by length (limited by model resources)
    df_filt = df_test[df_test["abstract"].str.len() < 4000]

    #
    ## Identify which keyword (disease and pathogen) is present in the abstract
    #

    # Process query to obtain disease and pathogens synonyms when used
    queries = np.where(
        df_filt['query'].str.contains('OR') == False,
            None,
            df_filt['query'].str.replace(" AND fha[FILT]", ""
                                        ).str.replace(
                                            f""" AND ("{"2004"}"[Date - Publication] : "{"2024/02/29"}"[Date - Publication])""",
                                            "").str.replace("[TIAB]", ""
                                                    ).str.replace("(", "").str.replace(")", ""))
    queries_split = [ query.split(" AND ") if query else [None, None] for query in queries ]
    keys = [(d.split(" OR ") , p.split(" OR ")) if d and p else (None, None) for d, p in queries_split]
    diseases = [[i.replace('"', "").lower() for i in d] if d else None for d, p in keys]
    pathogens = [[i.replace('"', "").lower() for i in p] if p else None for d, p in keys]

    # Infer which synonyms are in the abstract
    key_diseases = [[d for d in disease if (d in abstract or d in title)] if disease else None for disease, abstract, title in zip(diseases, 
                                                                                                                        list(df_filt['abstract_process']), 
                                                                                                                        list(df_filt.title_process))]
    key_pathogens = [[p for p in pathogen if (bacteria_regex_search(p, abstract) or bacteria_regex_search(p, title))] if pathogen else None for pathogen, abstract, title in zip(pathogens,
                                                                                                                       list(df_filt['abstract_process']), 
                                                                                                                       list(df_filt.title_process))]


    # Replace not matches by None
    key_diseases = [ get_longest(i) if (i and len(i) > 0) else None for i in key_diseases]
    key_pathogens = [ get_longest(i) if (i and len(i) > 0) else None for i in key_pathogens]

    # Stats
    correctly_matched = 0

    # find out which pathogens are correctly identified
    for key_pathogen, actual_disease, comment in zip(key_pathogens, df_reverence['Used Name'], df_reverence['Remark']) :
        if key_pathogen == None and (actual_disease == None or actual_disease == "None"): # No synonym was used and no synonym was added
            correctly_matched += 1
        elif comment == 'Less specific variant?': # key disease found is not the most specific
            ratio = fuzz.partial_ratio(key_pathogen, actual_disease)
            if ratio >= 90:
                correctly_matched += 1
        else:
            ratio_partial = fuzz.partial_ratio(key_pathogen, actual_disease)
            ratio_token_set = fuzz.token_set_ratio(key_pathogen, actual_disease)
            ratio = max(ratio_partial,  ratio_token_set)
            if ratio >= 95:
                correctly_matched += 1

    print(f"correctly matched diseases: {correctly_matched} / {len(key_pathogens)} = {correctly_matched / len(key_pathogens)}")


   


    
    
    