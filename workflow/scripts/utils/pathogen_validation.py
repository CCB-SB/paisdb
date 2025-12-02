import re
def existence_based_pathogen_validation(text, config):
    methods = config["methods"]

    methods = [r'(?:\W)' + f"(?P<method{pos}>{method.lower()})" + r'(?:\W)' for pos, method in enumerate(methods)]
    methods = r'|'.join(methods)
    found_methods = set()

    for match in re.finditer(methods, text.lower()):
        for _, methode in match.groupdict().items():
            if methode:
                found_methods.add(methode)
    
    return found_methods

def proximity_based_pathogen_validation(text, config):
    # load config
    pathogens = config["pathogen"]
    methods = config["methods"]
    distance = config["distance"]

    pathogen = f"(?:{'|'.join(pathogens)})"

    methods = [r'(?:\W)' + f"(?P<methode{pos}>{method.lower()})" + r'(?:\W)' for pos, method in enumerate(methods)]
    methods = '(?:' + '|'.join(methods) + ')'

    regex_pm = f"{pathogen.lower()}{r'.{1,'+ str(distance) + '}'}{methods}"
    regex_mp = f"{methods}{r'.{1,'+ str(distance) + '}'}{pathogen.lower()}"

    if distance == -1:
        regex_pm = f"{pathogen.lower()}{r'.{1,}'}{methods}"
        regex_mp = f"{methods}{r'.{1,}'}{pathogen.lower()}"
    
    found_methods = set()

    for match in re.finditer(regex_pm, text.lower()):
        for _, methode in match.groupdict().items():
            if methode:
                found_methods.add(methode)

    for match in re.finditer(regex_mp, text.lower()):
        for _, methode in match.groupdict().items():
            if methode:
                found_methods.add(methode)

    return found_methods

from spacy.tokens import Span
def sentence_co_occurrence_pathogen_validation(text, config):
    # load config
    pathogens = config['pathogen']
    methods = config['methods']
    nlp = config['nlp']

    methods = '(?:' + '|'.join(methods) + ')'

    regex = re.compile(methods, flags=re.IGNORECASE)

    Span.set_extension("contains_pathogen", getter=lambda span: any(pathogen.lower() in span.text.lower() for pathogen in pathogens), force=True)

    doc = nlp(text)
    methods = set()

    for sent in doc.sents:
        if sent._.contains_pathogen:
            matches = regex.finditer(sent.text)
            if matches:
                methods.update(match.group(0).strip() for match in matches)

    return methods


