#! /usr/bin/env/python
# coding: utf-8

## -----
# In-house entrezpy functions
# Author: G. Molano, LA (gonmola@hotmail.es)
# Last modified:
## -----

###########
# IMPORTS #
###########
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

from datetime import datetime

###########
# CLASSES #
###########

class MyMultiThreading_ncbi():
    def __init__(self, args_cmd, myfunc, file_tmp = "tmp.batch_0", threads=10, my_logger = None):
        self.args_cmd = args_cmd
        self.myfunc = myfunc
        self.file_tmp = file_tmp
        self.threads = threads
        self.my_logger = my_logger
    
    def run_batches_files(self, ranges_batches):
        """
        Run batches files using MultiThreading approach (I/O functions)
        """
        from concurrent.futures import ThreadPoolExecutor
        from utils.utils_concurrency import split_by_size
  
        max_retries = 10
        all_res =[]
        ranges_threads = split_by_size(input=len(ranges_batches), n=self.threads)
        self.my_logger.info(f"Ranges batches: len = {len(ranges_threads)}; [0] = {ranges_threads[0]}")
        
        for start, end in ranges_threads:
            attempt = 0
            success = False
            with ThreadPoolExecutor(self.threads) as pool:
                while not success and attempt <= max_retries:
                    try:
                        # issue all tasks to the thread pool
                        futures = [pool.submit(self.myfunc, j[0], j[1]) for j in self.args_cmd[start:end]]
                        # retrieve all return values in order
                        results = [future.result() for future in futures]
                        all_res.extend(results)

                        success = True
                        
                    except Exception as e:
                        self.my_logger.error(f"Something went wrong with Multithreading: {e}")
                        self.my_logger.error(f"Retrying iteration (attempt {attempt}): {start}-{end}")
                        success = False
                        attempt = attempt + 1
                    
            ## Check the stdder
            self.my_logger.info(f"Completed Tasks: {start} - {end}. Results = {len(results)}")
            [self.my_logger.debug(f"STDERR: {res[2]}\n CMD: {res[0]}") for  query_tup, res in results if res[2] if 'QUERY FAILURE' in res[2]]
        
        return all_res


###########
# FUNCTIONS #
###########    
def find_pattern(key, pattern, string, multiple=False):
    import re
    if key == 'abstract':
        # re.DOTALL to allow multiline search
        match = re.search(pattern, string, re.DOTALL)
    else:
        match = re.search(pattern, string)
    
    if match:
        if multiple:
            res = re.findall(pattern, string)
            res = [i.strip().replace('\n      ', "").strip() for i in res ]
            return res
        else:
            res = match.group(1)
            res = res.strip().replace('\n      ', "").strip()
            return res
    else:
        return None


def parse_medline_results(stdout):
    """
    Parse medline results from ncbi esummary function

    :params results: (list) STDOUT from esummary results
    :return (list) List of DocumentSummary objects
    """
    patterns = {
      'abstract': r'\nAB\s{0,2}-(.*?)\n[A-Z]',
      'pmid': r'PMID\s{0,2}-(.*)\n[A-Z]',
      'doi': r'\nLID\s{0,2}-(.*) \[doi\]', 
      'title': r'\nTI\s{0,2}-(.*)',
      'journal': r'\nJT\s{0,2}-(.*)',
      'publication_date_pubmed': r'\nPHST\s{0,2}-(.*) \[pubmed\]',
      'mesh_terms': r'\nMH\s{0,2}-(.*)',
      'substances': r'\nRN\s{0,2}-(.*)',
      'publication_type': r'\nPT\s{0,2}-(.*)',
      'pmc': r'\nPMC\s{0,2}-(.*)',
    }
    # Each data entry has to be splitted according to 'delim'
    delim = '\n\n'
    data = [(item, entry) for item in stdout for entry in item['results'].strip().split(delim) if entry]
    data_dict = {}
    for i, item in enumerate(data):
        info = item[0]
        entry = item[1]
        for key, patt in patterns.items():
            # Allow multiple matchings
            if key in ["mesh_terms", "susbtances", "publication_type"]:
                multiple=True
            else:
                multiple=False
            
            # Extract info
            data_dict.setdefault(i, {}).setdefault(key, find_pattern(key=key,
                                                                     pattern=patt,
                                                                     string=entry,
                                                                     multiple=multiple))
            data_dict[i]['disease'] = info['disease']
            data_dict[i]['pathogen'] = info['pathogen']
            data_dict[i]['query_key'] = info["query_key"]
            data_dict[i]['query'] = info['query']
    return data_dict