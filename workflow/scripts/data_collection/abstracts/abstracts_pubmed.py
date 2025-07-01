#!/usr/bin/env python
# coding: utf-8


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
from utils.utils import run_cmd
def make_ncbi_queries(query_tup, my_logger, date_collection_range=snakemake.params.date_collection_range):
    
    cmd = f"""esearch -db pubmed -query '{query_tup['query']}' | efetch -format medline"""
    # my_logger.info(f"CMD: {cmd}")
    res = run_cmd(cmd, False, True)
    
    # Retry without any synonyms
    if 'QUERY FAILURE' in res[2]:
        my_logger.info(f"FAILED (synonyms). Retry with only the original terms: {cmd}")
        query = f""" "{query_tup['disease']}"[TIAB] AND "{query_tup['pathogen']}"[TIAB] AND fha[FILT] AND ("{date_collection_range[0]}"[Date - Publication] : "{date_collection_range[1]}"[Date - Publication])"""
        cmd =  f"""esearch -db pubmed -query '{query}' | efetch -format medline"""
        my_logger.info(f"CMD Retry: {cmd}")
        res = run_cmd(cmd, False, True)
        query_tup['query'] = query
    
    return query_tup, res

##################################################
# MAIN
##################################################

if __name__ == '__main__':
    import os
    from utils.utils import run_cmd, get_api
    from utils.utils_concurrency import split_by_size
    from utils.utils_ncbi import MyMultiThreading_ncbi
    import pickle

    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "ncbi_request")
    my_logger = logger.get_logger()

    # Set NBCI API KEY
    os.environ["NCBI_API_KEY"] = get_api(
        api_file = snakemake.params.api_file, header = snakemake.params.ncbi_api)

    #
    ## Load batch pubmed query
    #
    queries_tup = pickle.load(open(snakemake.input.pickle, 'rb'))
    ranges_batches = split_by_size(input=len(queries_tup), n=snakemake.threads)
    my_logger.info(f"Requesting {len(queries_tup)} PUBMED searches in {len(ranges_batches)} batches using {snakemake.threads} threads...")

    # NOTE: Retriving a maximum of 10000 records per search (retmax = 10000)
    args_cmd = [(item, my_logger) for item in queries_tup]
    
    ## Print the first and last 2
    [my_logger.debug(f"Args: {args_cmd[i]}") for i in range(2)]
    [my_logger.debug(f"Args: {args_cmd[i]}") for i in range(len(args_cmd)-2,len(args_cmd))]

    #NOTE: only 10 threads each iteration to prevent too many requests
    ncbi_batches = MyMultiThreading_ncbi(
       args_cmd = args_cmd, myfunc = make_ncbi_queries,
       threads=snakemake.threads, my_logger = my_logger)
    results = ncbi_batches.run_batches_files(ranges_batches=queries_tup)
    
    my_logger.info(f"Results length: {len(results)}")
    pickle.dump(results, open(snakemake.output.pickle, 'wb'))