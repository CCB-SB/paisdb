##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))


##################################################
# FUNCTIONS
##################################################
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
    from scripts.utils.utils_my_logger import My_logger
    from rapidfuzz import fuzz
    from scripts.data_collection.pathogens.parse_ete4 import NCBITaxa_mod
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import re

    # Logging
    logger = My_logger(log_filename = 'tests/data_collection/relations/test.log', logger_name = "relation_filter_test")
    my_logger = logger.get_logger()

    # NCBI Taxonomy
    ETE4 = NCBITaxa_mod()
    ETE4.get_ncbi_names()

    relations = pd.read_csv("../workflow/tests/data_collection/relations/resources/relations_full.csv")

    my_logger.info(f"Input size: {relations.shape[0]}")

    # regex for unidentified
    regex = re.compile('unidentified')
    # Filter too broad pathogen assosiations
    highest_accepted_rank = 'order'
    my_logger.info(f"Accepted Rank: {highest_accepted_rank}")
    bool_list = []
    removed_list = []

    for pathogen in relations['pathogen']:
        if regex.search(pathogen):
            bool_list.append(False)
            removed_list.append('unidentified')
            continue
        lineage = ETE4.get_taxa_from_specie_taxid(sp_taxid=pathogen, taxid=False)
        if rank_to_int(lineage[2]) < rank_to_int(highest_accepted_rank):
            bool_list.append(False)
            removed_list.append(lineage[2])
        else:
            bool_list.append(True)

    removed = relations.loc[[not value for value in bool_list]]
    relations = relations.loc[bool_list]

    relations.to_csv("../workflow/tests/data_collection/relations/resources/relations_res.csv")
    removed.to_csv("../workflow/tests/data_collection/relations/resources/relations_rem.csv")

    my_logger.info(f"removed relations: {removed.shape[0]}")
    my_logger.info(f"accepted relations: {relations.shape[0]}")

    for reason in list(set(removed_list)):
        my_logger.info(f"{reason} removed relations: {removed_list.count(reason)}")


    