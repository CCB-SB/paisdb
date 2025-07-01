##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

import pandas as pd
import numpy as np

from utils.utils_my_logger import My_logger

##################################################
# Magic Numbers (Constants)
##################################################

uniqueness_base_score = 40
uniqueness_penalty = 10
validation_score = 10
ngs_score = 10

##################################################
# FUNCTIONS
##################################################

# find good scale for cohorts of size 1 to 100.000
def transform_cohort(cohort_size, logger):
    try:
        if cohort_size:
            if cohort_size < 100:
                return 10
            elif cohort_size <= 1000:
                return 20
            else:
                return 30
        else:
            return 0
    except:
        logger.info(f"{cohort_size}, {type(cohort_size)}")
        return -100

def transform_article_type(article_type):
    if article_type:
        if 'Case Reports' in article_type:
            return 5
        elif 'Study' in article_type:
            return 10
        elif 'Review' in article_type:
            return 20
        elif'Journal Article' in article_type:
            return 10
    return 0

def score_generator(cohort_size, validation, used_ngs, article_type, uniqueness, logger):
    score = 0
    score += transform_cohort(cohort_size, logger)
    score += validation_score if validation and validation != "Not available" else 0
    score += ngs_score if used_ngs and used_ngs != "Not available" else 0
    score += transform_article_type(article_type)
    score += uniqueness
    return score

##################################################
# MAIN
##################################################
if __name__ == '__main__':

    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "score generation")
    my_logger = logger.get_logger()

    # read file with extracted informations
    df = pd.read_csv(str(snakemake.input.csv), sep=",")
    df = df.replace({np.nan:None})

    # determine non-uniqueness for each PMID
    # non-uniqueness := Number of different pathogens for each disease

    non_uniqueness = []
    for pmid, pathogen, disease in zip(list(df['pmid']), list(df['pathogen']), list(df['disease'])):
        # select entrys with same pmid and disease
        df_sub = df[(df['pmid'] == pmid) & (df['disease'] == disease)]
        # non-uniqueness = min(0, base - penalty * (count - 1))
        non_uniqueness.append(max(0,uniqueness_base_score - uniqueness_penalty * (len(df_sub) - 1)))

    # apply score function for each entry
    score_list = []
    for cohort, validation, ngs, article_type, uniqueness in zip(list(df['cohort']), list(df['pathogen verification']), list(df['ngs']), list(df['publication_type']), non_uniqueness):
        score_list.append(score_generator(cohort, validation, ngs, article_type, uniqueness, my_logger))
    
    # save
    df['score'] = score_list
    df.to_csv(snakemake.output.csv, sep=",", index=False)