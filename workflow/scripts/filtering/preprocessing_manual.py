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

def get_longest(my_list):
    lengths = [len(i) for i in my_list]
    index = lengths.index(max(lengths))
    
    return my_list[index]



##################################################
# MAIN
##################################################
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from nltk.tokenize import sent_tokenize

    # Logging
    logger = My_logger(log_filename = "../results_old/results/filtering/pubmed_abstracts.log", logger_name = "ncbi_request")
    my_logger = logger.get_logger()

    df = pd.read_csv("../results_old/results/data_collection/abstracts/pubmed_abstracts.csv", index_col=0)
    # df.loc[:, 'publication_year'] = pd.to_datetime(df['publication_year']).dt.strftime('%Y')
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
    dispa['query_group'] = pd.cut(dispa.hits, bins=pd.IntervalIndex.from_breaks([0, 1, 100, max(dispa.hits)]))
    
    my_logger.info(f"# Pathogen-Disease groups {dispa.value_counts(subset='query_group', normalize=True)}")
    
    # Add group information
    df_merged = df.merge(dispa[['query_key', 'query_group']], on='query_key')

    # Filter abstract with > 1 abstracts 
    df_filt = df_merged[~df_merged.query_group.eq(pd.Interval(0,1, closed='right'))]
    my_logger.info(f"# Relations > 1 abstract: {len(df_filt.index)}")

    #
    ## Identify which keyword (disease and pathogen) is present in the abstract
    #
    
    # Process query to obtain disease and pathogens synonyms when used
    queries = np.where(
        df_filt['query'].str.contains('OR') == False,
            None,
            df_filt['query'].str.replace(" AND fha[FILT]", ""
                                        ).str.replace("[TIAB]", ""
                                                    ).str.replace("(", "").str.replace(")", ""))
    queries_split = [ query.split(" AND ") if query else [None, None] for query in queries ]
    keys = [(d.split(" OR ") , p.split(" OR ")) if p and d else (None, None) for d, p in queries_split]
    diseases = [[i.replace('"', "").lower() for i in d] if d else None for d, p in keys]
    pathogens = [[i.replace('"', "").lower() for i in p] if p else None for d, p in keys]
    
    # Stats
    dis_syns = [i for i in diseases if i]
    patho_syns = [i for i in pathogens if i]
    my_logger.info(f"# queries using disease synonyms: {len(dis_syns)}")
    my_logger.info(f"# queries using pathogen synonyms: {len(patho_syns)}")

    # Infer which synonyms are in the abstract
    key_diseases = [[d for d in disease if (d in abstract or d in title)] if disease else None for disease, abstract, title in zip(diseases, 
                                                                                                                        list(df_filt['abstract_process']), 
                                                                                                                        list(df_filt.title_process))]
    key_pathogens = [[p for p in pathogen if (p in abstract or p in title)] if pathogen else None for pathogen, abstract, title in zip(pathogens,
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
    df_filt['pmid'] = df_filt.loc[:, 'pmid'].astype('int64')
    df_filt.sort_values(by='pmid', inplace=True)

    #
    ## obtain pmid of processed relations
    #

    rela = pd.read_csv("../results/relation_extraction/benchmarking/result_1000.csv", index_col=0)
    cols_to_drop = ['Unnamed: 0', 'query_Mixtral-8x7B-Instruct-v0.1',
                    'query_gpt-3.5-turbo', 'answer_gpt-3.5-turbo',
                    'query_gpt-4', 'answer_gpt-4', 
                    'abstracts_filt', 'abstract_read',
                    'answer_Llama-2-70b-instruct-v2',
                    'answer_Llama-2-70b-instruct-v2_int',
                    'answer_gpt-3.5-turbo_int',
                    'answer_gpt-4_int']
    rela.drop(columns=cols_to_drop, inplace=True)

    # Filter by relationship
    rela['answer_Mixtral-8x7B-Instruct-v0.1_int'] = rela['answer_Mixtral-8x7B-Instruct-v0.1_int'].astype('int32')
    rela = rela[rela['answer_Mixtral-8x7B-Instruct-v0.1_int'] == 0]

    # TO DO: Modify colnames Relationships
    rela_cols = ['query_key', 'pmid', 
                 'answer_Mixtral-8x7B-Instruct-v0.1',
                 'answer_Mixtral-8x7B-Instruct-v0.1_int']
    rela_df = rela.loc[:, rela_cols].drop_duplicates()

    # Merge
    rela_new = rela_df.merge(df_filt, on=['query_key', 'pmid'], how='left').drop_duplicates()
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
    art_df = rela_new.loc[:, art_cols].drop_duplicates()
    art_df.to_csv("../src/webserver_draft_data/articles.csv", index=False)

    # Filter disease
    disease_df = pd.read_csv("../results/data_collection/diseases/diseases_full.csv")
    my_logger.info(f"# Diseases included in pubmed search: {len(disease_df.index)}")
    disease_df_filt = disease_df[disease_df.disease.isin(set(rela_new.disease))]
    my_logger.info(f"# Diseases with pathogen relations: {len(disease_df_filt.index)}")
    disease_df_filt = disease_df_filt.loc[:, ['disease_id', 'disease', 
                                              'definition','derives_from',
                                              'has_material_basis_in', 'has_symptom']]
    disease_df_filt.to_csv("../src/webserver_draft_data/diseases_full.csv", index=False)

    # Filter pathogens
    pathogens_df = pd.read_csv("../results/data_collection/pathogens/pathogens_full.csv")
    my_logger.info(f"# Pathogens included in pubmed search: {len(pathogens_df.index)}")
    pathogens_df_filt = pathogens_df[pathogens_df.pathogen.isin(set(rela_new.pathogen))]
    my_logger.info(f"# Pathogens with disease relation: {len(pathogens_df_filt.index)}")
    pathogens_df_filt.to_csv("../src/webserver_draft_data/pathogens_full.csv", index=False)
    
    

    # Merge diseases and pathogens

    # For relations
    rela_df = rela_df.merge(disease_df_filt[['disease', 'disease_id']], on='disease', how='left')
    rela_df = rela_df.merge(pathogens_df_filt[['pathogen', 'pathogen_id']], on='pathogen', how='left')
    rela_df.to_csv("../src/webserver_draft_data/relations.csv", index=False)

    # For merged table
    all_df = rela_df.merge(pathogens_df_filt[['pathogen_id',
                                            'superkingdom', 'phylum', 'class',
                                            'order','family', 'genus',
                                            'species']], on='pathogen_id',
                                            how='left')
    all_df.drop(columns = ['title','abstract',
       'publication_date_pubmed', 'substances'], inplace=True)
    all_df.to_csv("../src/webserver_draft_data/merged_table.csv", index=False)

    ##
    # Split df in batches
    ##
    ranges_batches = split_by_size(input=len(df.index), n=snakemake.params.batch_size)
    batches = [(batch, df.iloc[i[0]:i[1]]) for batch, i in enumerate(ranges_batches)]

    for batch, df_sub in batches:
        outfile = join(snakemake.output.DIR, f"batch_{batch}.csv")
        df_sub.to_csv(outfile, index=True)
    
    ##
    # Choose 1.000 samples
    ##

    # Include already read abstracts
    # df_85 = pd.read_csv("../results/filtering/preprocessed_abstracts_85.csv", index_col=0)
    # dfh = df_filt.copy(deep=True).reset_index()
    # df_85[['pathogen_term', 'disease_term']] = df_85.loc[:, ['pathogen_term', 'disease_term']].astype(str)
    # dfh[['pathogen_term', 'disease_term']] = dfh.loc[:, ['pathogen_term', 'disease_term']].astype(str)
    # read = df_85.merge(dfh, on=['pmid', 'pathogen_term', 'disease_term'])
    # not_include=list(read['index'])

    # N = 1000 - len(read.index)
    # df_filt_sub = df_filt.drop(not_include)
    # df_sub = df_filt_sub.sample(n=N, random_state = 123).reset_index(drop=True)

    
    # Split each abstracts into sentences 
    # rows = [(sent_tokenize(abstract), d, p ) for abstract, d, p in zip(list(df_sub['abstract']),
    #                                                        list(df_sub['pathogen_term']),
    #                                                        list(df_sub['disease_term']))]

    # Split each abstracts into sentences and only keep the sentence if keyword present
    # abstracts_filt = []
    # for index, row in df_sub.iterrows():
    #     filt = []
    #     for s in sent_tokenize(row['abstract']):
    #         patho_in_s = [True if i in s else False for i in row['pathogen_term']]
    #         disease_in_s = [True if i in s else False for i in row['disease_term']]

    #         if any(patho_in_s) or any(disease_in_s):
    #             filt.append(s)
        
    #     filt = ''.join(filt)
    #     abstracts_filt.append(filt)
    
    # df_sub.loc[:, 'abstracts_filt'] = abstracts_filt
    # df_sub.loc[:, 'abstracts_filt'] = abstracts_filt

    # # Capitalize abstract to enhance redability
    # abstracts_read = [' '.join([s.capitalize() for s in sent_tokenize(abstract)]) for abstract in list(df_sub['abstract'])]
    # df_sub['abstract_read'] = abstracts_read

    # # Save
    # df_sub.to_csv("../results/filtering/preprocessed_abstracts_957_all.csv", index=True)

    # # Reorder
    # all_cols = ['pmid', 'pathogen_term', 'disease_term', 'abstract_read', 
    #              'relationship', 'causality', 'controversy', 'model_article']
    # df_sub[all_cols[-4:]] = ""

    # read = read[all_cols]
    # df_sub[['pmid', 'pathogen_term',
    #         'disease_term', 'pathogen', 'disease',
    #          'query' ]].to_csv("../results/filtering/preprocessed_abstracts_957_info.csv", index=True)

    # # Append rows
    # df_1000 = pd.concat([df_sub[all_cols], read], axis=0)
    # df_1000.to_csv("../results/filtering/preprocessed_abstracts_1000.csv", index=True)
    
    
