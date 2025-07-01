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
    import pandas as pd

    df = pd.read_csv("../results/relation_extraction/benchmarking/result_1000.csv", index_col=0)
    cols_to_drop = ['Unnamed: 0', 'query_Mixtral-8x7B-Instruct-v0.1',
                    'query_gpt-3.5-turbo', 'answer_gpt-3.5-turbo',
                    'query_gpt-4', 'answer_gpt-4', 
                    'abstracts_filt', 'abstract_read',
                    'answer_Mixtral-8x7B-Instruct-v0.1', 'answer_gpt-3.5-turbo_int',
                    'answer_gpt-4_int','answer_Mixtral-8x7B-Instruct-v0.1_int']
    df.drop(columns=cols_to_drop, inplace=True)

    # Filter by relationship
    df['answer_Llama-2-70b-instruct-v2_int'] = df['answer_Llama-2-70b-instruct-v2_int'].astype('int32')
    df = df[df['answer_Llama-2-70b-instruct-v2_int'] == 0]

    # Relationships
    rela_cols = ['query_key', 'disease', 'pathogen',
                 'pmid', 'query_group', 
                 'disease_term', 'pathogen_term',
                 'answer_Llama-2-70b-instruct-v2',
                 'answer_Llama-2-70b-instruct-v2_int']
    rela_df = df.loc[:, rela_cols].drop_duplicates()
    rela_df.to_csv("../src/webserver_test_data/relations.csv", index=False)
    
    # Only articles
    art_cols = ['pmid', 'doi', 'title', 'abstract',
                'journal', 'publication_date_pubmed',
                'publication_year']
    art_df = df.loc[:, art_cols].drop_duplicates()
    art_df.to_csv("../src/webserver_test_data/articles.csv", index=False)
    
    