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

    df = pd.read_csv(snakemake.input.csv)
    df.drop(columns=["Taxonomy","Bio_rank","Organism_type"], inplace=True)
    df.rename(columns={'Pathogen name': 'pathogen'}, inplace=True)
    df['source']='gc'

    df.to_csv(snakemake.output.csv, index=False)



