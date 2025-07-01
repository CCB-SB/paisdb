#!/usr/bin/python

import re

######################################################
# API KEYS
######################################################
def get_api(api_file, header):
    from configparser import ConfigParser
    
    config = ConfigParser()
    config.read(api_file)
    API_KEY = config.get(header, 'api_key')

    return API_KEY
    

######################################################
# BASH RELATED
######################################################
def run_cmd(cmd, split = True, shell=True):
    import shlex

    """
    Run given CMD
    :param cmd: CMD (str)
    :param split: If the command has to be splitted. Set to False if the command has pipes '|' (bool)
    :param shell: If the command requires shell interpreter. Set to True if the command has pipes '|' (bool)
    :return: cmd (str), status (int), stdout + stderr (str)
    """
    import subprocess

    if split:
        cmd = shlex.split(cmd)

    # check = True, raise an exception if process fail
    try:
        p = subprocess.run(cmd, check=True, 
                        shell=shell, encoding="utf-8",
                        stdout = subprocess.PIPE,
                        stderr = subprocess.PIPE)

        p_stderr = p.stderr
        p_stdout = p.stdout
    
    except subprocess.CalledProcessError as exc:
        print(
            f"Process failed because did not return a successful return code. "
            f"Returned {exc.returncode}\n{exc} \n")
        raise exc

    return cmd, p_stdout, p_stderr

######################################################
# STRING RELATED
######################################################
def get_substr_between2chars(ch1,ch2,s):
    
    import re
    m = re.findall(ch1+'(.+?)'+ch2, s)
    if m:
        m = [i.strip() for i in m]
        return m

def process_string(s):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    # Remove non-alphanumeric characters
    s = s.lower().strip().replace('"', '').replace("'", "")
    clean_string = []
    for word in s.split(" "):
        e = ''.join(e for e in word if e.isalnum() or e != "-")
        if e != "":
            clean_string.append(e)
    s = " ".join(clean_string)
    
    # prevent stopwords
    querywords = s.split()
    resultwords = [word for word in querywords if word.lower() not in stopwords.words('english')]
    s = ' '.join(resultwords)
    s = s.lower().strip()

    # Tokenize + normalize (lemmatization = remove affixes if the resulting word is in its dictionary)
    wnl = wnl = nltk.WordNetLemmatizer()
    tokens = [token.lower() for token in word_tokenize(s)]
    lemmatized_words = [wnl.lemmatize(token) for token in tokens]
    s = " ".join(lemmatized_words)

    return s

def fuzzy_search(query, choice_list, scorer, threshold=90.1):
    from rapidfuzz import process
    from rapidfuzz import fuzz
    """
    Make a fuzzy search 
    :param query: search disease in ont
    :param choice_list: list of all strings the query should be compared
    :param threshold: minimal similarity for the matches
    :return: tupel, where first is the similarity score and second pos. is the dict entry
    """
    # Lower case both query and choice_list
    if query == 'aids':
        print(choice_list)
    choice_list_low = [i.lower() for i in choice_list]

    res = process.extractOne(query.lower(), choice_list_low, score_cutoff=threshold, scorer=scorer)
    if res:
        index = choice_list_low.index(res[0])

        if scorer == fuzz.WRatio:
            best_match = {'score': res[1], 'match': choice_list[index], 'method': 'wratio'}
        elif scorer == fuzz.token_set_ratio:
            best_match = {'score': res[1], 'match': choice_list[index], 'method': 'token_set_ratio'}
        else:
            raise KeyError("Invalid scorer. Please choose between fuzz.WRatio and fuzz.token_set_ratio")

        return best_match