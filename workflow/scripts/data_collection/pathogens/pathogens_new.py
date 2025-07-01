##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

##################################################
# FUNCTIONS
##################################################
def map_names(name):
    try:
        res = ETE4.get_taxa_from_specie_taxid(sp_taxid=name, taxid=False,
                append_ID=True)
        name = res[-1]
        id = res[0][-1]
        return {'name': name, 'id': id}
    except ValueError as e:
        return None

def map_taxonomy(taxid):
    try:
        res = ETE4.get_taxa_from_specie_taxid(sp_taxid=taxid, taxid=True,
                append_ID=True)
        return res
    except ValueError as e:
        return None

def rank_to_int(rank):
    mapping = {
        'superkingdom': 0, 
        'phylum': 1, 
        'class': 2, 
        'order': 3, 
        'family': 4, 
        'genus': 5, 
        'species': 6
    }
    return mapping.get(rank)

##################################################
# MAIN
##################################################
if __name__ == "__main__":
    from utils.utils_my_logger import My_logger
    from rapidfuzz import fuzz
    from parse_ete4 import NCBITaxa_mod
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import re

    #import nltk
    #nltk.download('punkt')
    #nltk.download('wordnet')

    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "pathogen_terms")
    my_logger = logger.get_logger()

    # NCBI Taxonomy
    ETE4 = NCBITaxa_mod()
    ETE4.get_ncbi_names()

    # Load PAISDB pathogens
    if snakemake.params.paisdb:
        pais_df = pd.read_csv(snakemake.params.paisdb, header=0)
        pais_pathogens = list(pais_df['pathogen'].str.lower())
        my_logger.info(f"PAIS DB (v.{set(pais_df['version'])}) = {len(pais_df.index)} pathogen names")
    else:
        my_logger.info(f"NO PAIS DB pathogen names provided. All entries will be considered as new.")
        pais_pathogens = set()
    
    # New entries
    new_df = pd.read_csv(snakemake.input.new_entries, header=0)
    names = list(new_df['pathogen'])
    my_logger.info(f"Entries in new_entries df = {len(names)}")
    new_entries = [name for name in names if name not in pais_pathogens]
    my_logger.info(f"Preprocessing and mapping possible {len(set(new_entries))} new pathogen names using NCBI Taxonomy")

    ##
    # Normalize pathogen names using NCBI Taxonomy
    ##
    
    patho_map = {}
    patho_map_id = {}
    for i in tqdm(set(new_entries)):
        res1 = map_names(i)
        
        if res1:
            patho_map.update({i:res1['name']})
            patho_map_id.update({res1['name']: res1['id']})
        else:
            res2 = ETE4.get_matching_node(query = i,
                                scorer=fuzz.WRatio,
                                threshold=90.1)
            if res2 and res2['match'] != '':
                patho_map.update({i:res2['term']})
                patho_map_id.update({res2['term']: res2['ID']})
    
    my_logger.info(f"Normalize pathogen names ({len(set(new_entries))}) = {len(patho_map.keys())} matches [NCBI Taxonomy] ")
    new_df['pathogen'] = new_df.loc[:, 'pathogen'].replace(patho_map)

    # Add taxID
    new_df['pathogen_id'] = new_df.loc[:,'pathogen']
    new_df['pathogen_id'] = new_df.loc[:, 'pathogen_id'].replace(patho_map_id)
    new_df['pathogen_id'] = np.where(new_df['pathogen_id'].isin(patho_map_id.values()),
                                        new_df['pathogen_id'], None)
    # Delete pathogens without taxid
    new_df = new_df.loc[~new_df['pathogen_id'].isna()]

    
    ##
    # Normalize pathogen names from relations
    ##

    relations = pd.read_csv(snakemake.input.relations)
    relations['pathogen'] = relations.loc[:, 'pathogen'].replace(patho_map)
    
    # Add taxID
    relations['pathogen_id'] = relations.loc[:,'pathogen']
    relations['pathogen_id'] = relations.loc[:, 'pathogen_id'].replace(patho_map_id)
    relations['pathogen_id'] = np.where(relations['pathogen_id'].isin(patho_map_id.values()),
                                        relations['pathogen_id'], None)
    # Delete pathogens without taxid
    relations = relations.loc[~relations['pathogen_id'].isna()]
    relations.drop(columns='pathogen_id', inplace=True)
    
    # Remove duplicates, preferece DO-Wikipedia
    relations.sort_values(by=['source'], inplace=True)
    relations.drop_duplicates(subset=['disease', 'pathogen', 'pathogen_type'], keep='first', inplace=True)

    # Sort and Save
    relations.sort_values(by=['disease', 'pathogen', 'pathogen_type', 'source'], inplace=True)
    relations.to_csv(snakemake.output.relations_csv, index=False)

    # relations
    pathogens_cause_disease = set(relations.loc[relations['pathogen_type'] == 'cause']['pathogen'])
    pathogens_asso_disease = set(relations.loc[relations['pathogen_type'] == 'asso']['pathogen'])
    disease_with_pathogen = set(relations.loc[relations['pathogen_type'] == 'cause']['disease'])

    ##
    # Add Relation information to diseases
    ##

    diseases = pd.read_csv(snakemake.input.diseases)
    diseases['caused_by_pathogen'] = np.where(diseases['disease'].isin(disease_with_pathogen),
                                            True, False)
    
    do_terms = pd.read_csv(snakemake.input.do_terms)
    diseases = diseases.merge(do_terms, how='left', left_on='disease', right_on='name')

    diseases.to_csv(snakemake.output.diseases_csv, index=False)
    
    ##
    # Add Relation information to pathogens
    ##

    new_df['disease_cause'] = np.where(new_df['pathogen'].isin(pathogens_cause_disease),
                                            True, False)
    new_df['disease_asso'] = np.where(new_df['pathogen'].isin(pathogens_asso_disease),
                                            True, False)
    
    # Retrieve lineage and synonyms for new pathogens
    taxonomy = [ map_taxonomy(taxid) for taxid in new_df['pathogen_id']]

    # retrieve syns for ID in taxids
    dict_data = {}
    extra = []
    for item in taxonomy:

        lineage = item[0]
        lineage_id = item[1]
        last_rank = item[2]
        last_name_rank = item[3]

        taxid = lineage[-1]
        dict_data[taxid] =  {
            'pathogen_id': taxid, 'lowest_rank': last_rank,
            'superkingdom': lineage[0], 'phylum': lineage[1], 'class': lineage[2],
            'order': lineage[3],'family': lineage[4],'genus': lineage[5],
            'species': lineage[6],
            'superkingdom_id': lineage_id[0], 'phylum_id': lineage_id[1],
            'class_id': lineage_id[2], 'order_id': lineage_id[3],
            'family_id': lineage_id[4], 'genus_id': lineage_id[5],
            'species_id': lineage_id[6] }
        
        syns = list()
        if taxid in ETE4.db_terms['common']:
            syns.extend(ETE4.db_terms['common'][taxid])
        if taxid in ETE4.db_terms['syn']:
            syns.extend(ETE4.db_terms['syn'][taxid])
        
        
        syns = list(set(re.sub(r'\(.*\)', '', term) for term in syns)) # Remove parenthesis
        syns = list(set(term.split(',')[0] for term in syns)) # Do not include things after comma
        syns = list(set(term.replace('"', "'") for term in syns)) # Replace Double quotes by single ones
        
        
        string = '|'.join(syns)
        dict_data[taxid].setdefault('synonyms', string.strip())

    df_lineage = pd.DataFrame.from_dict(dict_data, orient='index')
    new_df = new_df.merge(df_lineage, on="pathogen_id", how='left')

    # Remove duplicates and sort
    new_df.sort_values(by=['pathogen', 'source'], inplace=True)
    new_df.drop_duplicates(subset=['pathogen'], keep='first', inplace=True)
    new_df.sort_values(by=['pathogen'], inplace=True)

    # regex for unidentified
    regex = re.compile('unidentified')
    # Filter too broad pathogen assosiations
    highest_accepted_rank = 'genus'
    bool_list = []

    for pathogen, lowest_rank in zip(list(new_df['pathogen']),list(new_df['lowest_rank'])):
        if not lowest_rank:
            my_logger.info(f"No rank for pathogen {pathogen}")
            bool_list.append(False)
            continue
        if regex.search(pathogen):
            bool_list.append(False)
            continue
        if rank_to_int(lowest_rank) < rank_to_int(highest_accepted_rank):
            bool_list.append(False)
        else:
            bool_list.append(True)
    new_df = new_df.loc[bool_list]

    # Save
    new_df.to_csv(snakemake.output.pathogens_csv, index=False)
    



    
