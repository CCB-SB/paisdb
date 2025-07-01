##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

from utils.utils_my_logger import My_logger
from utils.utils_concurrency import split_by_size


import pandas as pd
from bioc import biocxml
from bioc import BioCCollection
import json
import requests
import time

##################################################
# FUNCTIONS
##################################################


##################################################
# MAIN
##################################################
if __name__ == '__main__':

    # Logging
    logger = My_logger(log_filename = snakemake.log[0], logger_name = "pmc_article_retrieving")
    my_logger = logger.get_logger()

    # keep articles where the LLM found a relationship
    abstract_csv = pd.read_csv(str(snakemake.input.csv))
    abstract_csv['pmid'] = abstract_csv['pmid'].astype("str")
    abstract_csv = abstract_csv.loc[abstract_csv['relation-extraction'] == 1.0]
    my_logger.info(f"Found articles with relationships: {len(abstract_csv)}")

    # sort by pmid
    abstract_csv.sort_values('pmid')

    unique_pmids = list(set(abstract_csv['pmid']))
    pmid_batches = split_by_size(input=len(unique_pmids), n=snakemake.params.batch_size)

    article_with_pmc = {}
    error_handle = False

    #initial xml file
    i = pmid_batches[0]
    pmids = ",".join(unique_pmids[i[0]:i[1]])
    response = requests.get(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmids}/unicode")

    try:
        collection = biocxml.loads(response.text)
        time.sleep(2)

        # patching other batches in if they exist
        for i in pmid_batches[1:]:
            pmids = ",".join(unique_pmids[i[0]:i[1]])
            response = requests.get(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmids}/unicode")
            p_collection = biocxml.loads(response.text)
            for document in p_collection.documents:
                collection.documents.append(document)
            time.sleep(2)

        # retrieving PMIDs by looking at the document information
        for doc in collection.documents:
            # Extract PMID, because Document ID is PMCID
            info = doc.passages[0].infons
            if "article-id_pmid" in info:
                article_with_pmc[doc.passages[0].infons['article-id_pmid']] = doc.id

            else:
                # lookup PMID in csv to find corresponding PMID
                PMID = abstract_csv[ abstract_csv['pmc'] == doc.id ]['pmid'].unique()
                if len(PMID) > 0:
                    article_with_pmc[PMID[0]] = doc.id
    except Exception as e:
        my_logger.info(f"Error while requesting pmc article,{e}")
        error_handle = True

    # We include articles which only have abstracts as well
    # filter abstract csv to only contain entries which are articles with pmcid
    # abstract_csv = abstract_csv[abstract_csv['pmid'].isin(article_with_pmc.keys())]

    # add PMCID to abstracts csv
    abstract_csv['pmc'] = [article_with_pmc[pmid] if pmid in article_with_pmc else None for pmid in list(abstract_csv['pmid'])]
    abstract_csv['pmc'] = [(pmc if 'PMC' in pmc else "PMC"+ pmc) if pmc else None for pmc in abstract_csv['pmc']]

    # Create Outputs
    abstract_csv.to_csv(snakemake.output.csv, index=False)
    # potentially large XML file containing the full articles on the document level
    with open(snakemake.output.xml, "w") as f:
        if not error_handle:
            biocxml.dump(collection, f)
        else:
            collection = BioCCollection()
            collection.source = "No search results found"
            biocxml.dump(collection, f)
        
    my_logger.info(f"Found articles with PMCid: {len(abstract_csv[abstract_csv['pmid'].isin(article_with_pmc.keys())])}")