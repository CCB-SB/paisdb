##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

##################################################
# FUNCTIONS
##################################################

##################################################
# MAIN
##################################################
if __name__ == "__main__":
    from utils.utils_my_logger import My_logger
    from rapidfuzz import fuzz
    from parse_ontology import Disease_ontology
    import pandas as pd
    import numpy as np
    from pprint import pprint


    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "disease_terms")
    my_logger = logger.get_logger()

    
    # Load Disease Ontology information
    my_logger.info(f"Retrieving and processing Disease Ontology information: {snakemake.params.disease_ont}")
    DO = Disease_ontology(location="doid.obo")
    DO.build_graph_doid()
    my_logger.info(f"Using {DO.name} version {DO.version} ({DO.url})")

    # Load PAISDB diseases
    if snakemake.params.paisdb:
        pais_df = pd.read_csv(snakemake.params.paisdb, header=0)
        pais_diseases = list(pais_df['disease'].str.lower())
        my_logger.info(f"PAIS DB (v.{set(pais_df['version'])}) = {len(pais_df.index)} disease names")
    else:
        my_logger.info(f"NO PAIS DB diseases names provided. All entries will be considered as new.")
        pais_diseases = set()
    
    # New entries
    new_df = pd.read_csv(snakemake.input.new_entries, header=0)
    names = set(new_df['disease'])
    my_logger.info(f"Entries in new_entries df = {len(names)}")
    new_entries = [entry for entry in names if entry not in pais_diseases]
    my_logger.info(f"Preprocessing and mapping possible {len(new_entries)} new disease names to {DO.name}")

    # Map to HUman Disease Ontology
    disease_map = {item: DO.get_matching_node(query=item,
                                              threshold = snakemake.params.threshold_wratio_do,
                                              scorer = fuzz.WRatio,
                                              string_process=False
                                           ) for item in new_entries if not pd.isna(item)}
    # Unmatches
    unmatches = {key for key, value in disease_map.items() if not value or value['match'] == ''}
    my_logger.info(f'# unmatches = {len(unmatches)}')
    unmatches_df = pd.DataFrame(list(unmatches), columns=['disease'])
    unmatches_df.sort_values(by='disease', inplace=True)
    unmatches_df.to_csv(snakemake.output.unmatches, index=False)
    
    # Matches
    disease_map_all = {key:value for key, value in disease_map.items() if value if value['match'] != ''}
    disease_map = {key:value['term'] for key, value in disease_map_all.items()}
    disease_map_id = {value['term']:value['term_id'] for key, value in disease_map_all.items() if value if value['match'] != ''}
    my_logger.info(f"""
                   Normalizing disease terms ({len(new_entries)}): 
                        {len(disease_map.keys())} matches [DO weighted ratio (>={snakemake.params.threshold_wratio_do} similarity)]
                   """)

    # Replace disease term
    new_df['disease'] = new_df.loc[:, 'disease'].replace(disease_map)
    # Add diseaseID
    new_df['disease_id'] = new_df.loc[:,'disease']
    new_df['disease_id'] = new_df.loc[:, 'disease_id'].replace(disease_map_id)
    new_df['disease_id'] = np.where(new_df['disease_id'].isin(disease_map_id.values()),
                                           new_df['disease_id'], None)
    print(len(new_df.index))

    # Add synonyms
    syns = {name: {'synonyms': 
                   list( DO.relation_dict[name]["has_exact_synonym"].union(
                       DO.relation_dict[name]["has_related_synonym"])
                       )
                    } for name in set(disease_map.values()) }
    syns = { key: {'synonyms': '|'.join([s.replace('"', "'") for s in item['synonyms']])} for key, item in syns.items()} # Replace Double quotes by single ones
    syns_df = pd.DataFrame.from_dict(syns, orient='index').reset_index(names='disease')
    new_df = new_df.merge(syns_df, on='disease', how='left')
    print(new_df.head())

    # Check again if new entries
    names = new_df['disease']
    new_entries = [entry for entry in names if entry not in set(pais_diseases)]
    pf_filt = new_df.loc[new_df['disease'].isin(new_entries)]
    
    # Drop duplicates and order
    pf_filt.sort_values(by=['disease', 'source'], inplace=True)
    pf_filt.drop_duplicates(subset=['disease'], keep='first',  inplace=True)
    pf_filt.sort_values(by='disease', inplace=True)
    my_logger.info(f"New diseases entries: {len(pf_filt.index)} )")
    
    pf_filt.to_csv(snakemake.output.diseases_csv, index=False)

    # Pathogens related to diseases
    pathogens_do = { i:{
        'pathogen': DO.relation_dict[i]['has_material_basis_in'],
        'disease_id': DO.relation_dict[i]['id']
        } for i in set(disease_map.values()) if 'has_material_basis_in' in DO.relation_dict[i]}
    
    # To long format
    df = pd.DataFrame.from_dict(pathogens_do, orient='index').reset_index(names='disease')
    rows = []
    for index, row in df.iterrows():
        if row['pathogen']:
            for patho in row['pathogen']:
                rows.append((row['disease'], patho, 'cause', 'DO'))
    pathos_do = pd.DataFrame(rows, columns = ['disease', 'pathogen', 'pathogen_type', 'source'])
    my_logger.info(f"Pathogens with 'has_basis_in' do diseases: {len(set(pathos_do['disease']))} diseases ; {len(set(pathos_do['pathogen']))} causative agents")
    
    # Relations
    pathos_do.to_csv(snakemake.output.relations_csv, index=False)
    # Pathogens
    pathos_df = pathos_do.loc[:, ['pathogen', 'source']].drop_duplicates()
    pathos_df.to_csv(snakemake.output.pathogens_csv, index=False)
    
    # DO information
    do_df = pd.DataFrame.from_dict(DO.relation_dict, orient='index')
    do_df.to_csv(snakemake.output.do_terms, index=False)

    