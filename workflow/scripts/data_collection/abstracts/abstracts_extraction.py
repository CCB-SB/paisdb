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
    from utils.utils_ncbi import parse_medline_results
    import pickle
    
    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "ncbi_request")
    my_logger = logger.get_logger()

    results = pickle.load(open(str(snakemake.input.pickle), 'rb'))
    my_logger.info(f"Results length: {len(results)}")

    results_queries = [{'results': res[1],
                        'disease': query['disease'],
                        'pathogen': query['pathogen'],
                        'query_key': query['keywords'],
                        'query': query['query']} for query, res in results if res[1]]
    my_logger.info(f"No empty results-queries: {len(results_queries)}")

    # Parse results in medline format.
    data_dict = parse_medline_results(stdout=results_queries)
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.loc[:, 'publication_date_pubmed'] = pd.to_datetime(df['publication_date_pubmed'], format = '%Y/%m/%d %H:%M').dt.strftime('%Y/%m/%d')
    df.loc[:, 'publication_year'] = pd.to_datetime(df['publication_date_pubmed']).dt.strftime('%Y')
    df.dropna(subset='abstract', inplace=True)
    df.to_csv(snakemake.output.csv, index=True)
    
    