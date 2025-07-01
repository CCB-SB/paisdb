##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

from utils.utils_my_logger import My_logger
##################################################
# FUNCTIONS
##################################################



##################################################
# MAIN
##################################################
if __name__ == '__main__':
    import pandas as pd
    from glob import glob


    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "data_tables")
    my_logger = logger.get_logger()

    # Load relation extraction batches
   #  dfs = [pd.read_csv(i, index_col=0) for i in glob("../results/relation_extraction/mixtral/batches_2004-20240229/batch_*.csv")]
    dfs = [pd.read_csv(i, index_col=0) for i in snakemake.input.relations]
    rela = pd.concat(dfs)
    rela.drop(columns='query_Mixtral-8x7B-Instruct-v0.1', inplace=True)
    my_logger.info(f"# Mixtral results: {len(rela.index)}")

    # Filter by relationship
    rela['answer_Mixtral-8x7B-Instruct-v0.1_int'] = rela['answer_Mixtral-8x7B-Instruct-v0.1_int'].astype('int32')
    rela = rela[rela['answer_Mixtral-8x7B-Instruct-v0.1_int'] == 1]
    my_logger.info(f"# Relationships (mitraxl results marks as relationship): {len(rela.index)}")

    # Load articles batches
   #  dfs = [pd.read_csv(i) for i in glob("../results/filtering/abstracts_2004-20240229/batch_*.csv")]
    dfs = [pd.read_csv(i) for i in snakemake.input.articles]
    arts = pd.concat(dfs)

    # Keep only articles with significant relationship
    rela_new = rela.merge(arts, on=['query_key', 'pmid'], how='left').drop_duplicates()
    rela_df = rela_new.reset_index(names='relation_id')
    rela_df = rela_df.loc[:, ['relation_id', 'disease',
       'pathogen', 'query', 
       'pmid', 'title','abstract', 'journal',
       'publication_date_pubmed', 'publication_year',
       'publication_type', 
       'mesh_terms', 'substances']]

    # Only articles
    art_cols = ['pmid', 'title', 'abstract',
                'journal', 'publication_date_pubmed',
                'publication_year', 'publication_type',
                'mesh_terms', 'substances']
    art_df = rela_df.loc[:, art_cols].drop_duplicates()
    art_df.to_csv(snakemake.output.articles, index=False)
    my_logger.info(f"# Articles: {len(art_df.index)}")

    # Merge

    # Filter disease
    disease_df = pd.read_csv(snakemake.input.diseases)
    my_logger.info(f"# Diseases included in pubmed search: {len(disease_df.index)}")
    disease_df_filt = disease_df[disease_df.disease.isin(set(rela_new.disease))]
    my_logger.info(f"# Diseases with pathogen relations: {len(disease_df_filt.index)}")
    disease_df_filt = disease_df_filt.loc[:, ['disease_id', 'disease', 
                                              'definition','derives_from',
                                              'has_material_basis_in', 'has_symptom']]
    disease_df_filt.to_csv(snakemake.output.diseases, index=False)

    # Filter pathogens
    pathogens_df = pd.read_csv(snakemake.input.pathogens)
    my_logger.info(f"# Pathogens included in pubmed search: {len(pathogens_df.index)}")
    pathogens_df_filt = pathogens_df[pathogens_df.pathogen.isin(set(rela_new.pathogen))]
    my_logger.info(f"# Pathogens with disease relation: {len(pathogens_df_filt.index)}")
    pathogens_df_filt.to_csv(snakemake.output.pathogens, index=False)
    
    

    # Merge diseases and pathogens

    # For relations
    rela_df = rela_df.merge(disease_df_filt[['disease', 'disease_id']], on='disease', how='left')
    rela_df = rela_df.merge(pathogens_df_filt[['pathogen', 'pathogen_id']], on='pathogen', how='left')
    rela_df.to_csv(snakemake.output.relations, index=False)

    # For merged table
    all_df = rela_df.merge(pathogens_df_filt[['pathogen_id',
                                            'superkingdom', 'phylum', 'class',
                                            'order','family', 'genus',
                                            'species']], on='pathogen_id',
                                            how='left')
    all_df.drop(columns = ['title','abstract',
       'publication_date_pubmed', 'substances'], inplace=True)
    all_df.drop_duplicates(inplace=True)
    all_df.to_csv(snakemake.output.merged, index=False)
