##################################################
# IMPORTS
##################################################
import re
import pandas as pd


##################################################
# FUNCTIONS
##################################################
'''
    Given a Match extract which group the match comes from
'''
def handle_groups(match):
    if match.group(1):
        return match.group(1)
    elif match.group(2):
        return match.group(2)
    elif match.group(3):
        return match.group(3)
    elif match.group(4)
        return match.group(4)
    else:
        return ''

##################################################
# MAIN
##################################################
if __name__ == '__main__':

    # Patterns
    nc_id = "(nc[0-9]{1,5})"
    someChars_strain = r" ((?:[A-Za-z]{1,6}[0-9]{1,4}){1,3})(?: and|,| or)(?=.{0,30} strains?)| ((?:[A-Za-z]{1,6}[0-9]{1,4}){1,3})\)? strain"
    atcc_number = r"(atcc [0-9]{3,7}|ATCC [0-9]{3,7})"

    # unused patterns
        #strain_number = "strain [a-z]{0,3}[0-9]{1,6}"
        #number_strain = r" ((?:(-|\/|\w){2,20}) strain)" # to broad
        #st_id = r" st[0-9]{1,3} | \(st[0-9]{1,3}\)" # might be possible but test set non
        #other_strain_1 = "[a-z][0-9]:[a-z][0-9]{1,4}"
        #short_strain = "[A-Z][a-z]{1,2}[0-9]{1,3}"

    # Final Pattern 
    pattern = re.compile(f"{number_strain_1}|{nc_id}|{atcc_number}")

    # load articles
    articles = pd.read_csv(str(snakemake.input.csv))

    # Determine strain
    strain_info = []
    for i, abstract in enumerate(articles['abstract']):
        matches = [handle_groups(match) for match in pattern.finditer(abstract) if handle_groups(match) != '']
        if len(matches) > 0:
            strain_info.append(matches)
        else:
            strain_info.append(None)
    articles['strain_info'] = strain_info

    # save csv
    articles.to_csv(snakemake.output.csv, index=False)
    

