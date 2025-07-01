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

    #
    ## Wikipedia list
    #

    dp_df = pd.read_csv(snakemake.input.csv, na_values=None)
    dp_df.drop(columns=['Links', 'Source'], inplace=True)

    # Add pathogen_type
    dp_df['index'] = dp_df.index
    dp_df = pd.wide_to_long(dp_df, stubnames='pathogen', i='index', j='pathogen_type', 
                    sep='_', suffix=r'\w+').reset_index()
    dp_df.drop(columns='index', inplace=True)


    # One line for each pathogen-pathogen_type-disease: easy for eliminating duplicates
    pathos = [i.split(', ') if not pd.isna(i) else None for i in list(dp_df['pathogen'])]
    dp_df['pathogen'] = pathos
    
    rows = []
    for index, row in dp_df.iterrows():
        if row['pathogen']:
            for patho in row['pathogen']:
                rows.append((row['disease'].strip(), patho.strip(), row['pathogen_type'], 'wikipedia'))
    wiki_pathos = pd.DataFrame(rows, columns = ['disease', 'pathogen', 'pathogen_type', 'source'])

    # Deduplicate, prioritazing causative
    wiki_pathos.sort_values(by=['disease', 'pathogen', 'pathogen_type'], inplace=True)
    wiki_pathos.drop_duplicates(subset=['disease', 'pathogen'], keep='last', inplace=True)
    
    # Relations
    wiki_pathos.to_csv(snakemake.output.relations_csv, index=False)
    
    # Diseases 
    disease_df = wiki_pathos.loc[:, ['disease', 'source']].drop_duplicates()
    disease_df.to_csv(snakemake.output.diseases_csv, index=False)
    
    # Pathogens 
    pathos_df = wiki_pathos.loc[:, ['pathogen','source']].drop_duplicates()
    pathos_df.to_csv(snakemake.output.pathogens_csv, index=False)
