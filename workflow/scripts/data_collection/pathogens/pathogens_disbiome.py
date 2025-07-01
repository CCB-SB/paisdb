
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

    disbiome = {}
    for index, i in enumerate(data):
        disbiome.setdefault(index, {}
                           ).setdefault('disease', i['disease_name'])
        disbiome[index]['pathogen'] = i['organism_name']

    patho_df = pd.DataFrame.from_dict(disbiome, orient='index')
    patho_df['pathogen_type'] = "asso"
    patho_df['source'] = "disbiome"
    

    # Relations
    patho_df.to_csv(snakemake.output.relations_csv, index=False)

    # Pathogens 
    pathos_df = patho_df.loc[:, ['pathogen','source']].drop_duplicates()
    pathos_df.to_csv(snakemake.output.pathogens_csv, index=False)

    # Diseases 
    disease_df = patho_df.loc[:, ['disease', 'source']].drop_duplicates()
    disease_df.to_csv(snakemake.output.diseases_csv, index=False)