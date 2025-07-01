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

    import json

    #
    ## PathoDB
    #

    with open(snakemake.input.json, 'r') as f:
        data = json.load(f)

    pathodb = {}
    for index, i in enumerate(data['results']['bindings']):
        pathodb.setdefault(index, {}
                           ).setdefault('disease', i['Disease']['value'])
        pathodb[index]['pathogen'] = i['Pathogen']['value']

    patho_df = pd.DataFrame.from_dict(pathodb, orient='index')
    patho_df['source'] = "pathodb"
    

    # Relations
    patho_df['pathogen_type'] = 'cause'
    patho_df.to_csv(snakemake.output.relations_csv, index=False)

    # Pathogens 
    pathos_df = patho_df.loc[:, ['pathogen','source']].drop_duplicates()
    pathos_df.to_csv(snakemake.output.pathogens_csv, index=False)