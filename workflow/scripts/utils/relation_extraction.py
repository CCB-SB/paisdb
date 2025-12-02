
import re
def existence_based_relationship(text, config):
    # load config
    pathogens = config['pathogen']
    diseases = config['disease']
    pathogen = '|'.join(pathogens)
    disease = '|'.join(diseases)
    pathogen_match = re.search(pathogen.lower(), text.lower())
    disease_match = re.search(disease.lower(), text.lower())
    return 'yes' if pathogen_match and disease_match else 'no'



def proximity_based_relationship(text, config):
    # load config
    pathogens = config['pathogen']
    diseases = config['disease']
    distance = config['distance']

    pathogen = f"(?:{'|'.join(pathogens)})"
    disease = f"(?:{'|'.join(diseases)})"

    regex = f"({pathogen.lower()}{r'.{1,'+ str(distance) + '}'}{disease.lower()})|({disease.lower()}{r'.{1,'+ str(distance) + '}'}{pathogen.lower()})"

    if distance == -1:
        regex = f"({pathogen.lower()}{r'.{1,}'}{disease.lower()})|({disease.lower()}{r'.{1,}'}{pathogen.lower()})"
    match = re.search(regex, text.lower())
    return 'yes' if match else 'no'


from spacy.tokens import Span
def sentence_based_relationship(text,config):
    # load config
    nlp = config['nlp']
    pathogens = config['pathogen']
    diseases = config['disease']

    Span.set_extension("contains_pathogen", getter=lambda span: any(pathogen.lower() in span.text.lower() for pathogen in pathogens), force=True)
    Span.set_extension("contains_disease", getter=lambda span: any(disease.lower() in span.text.lower() for disease in diseases), force=True)

    doc = nlp(text)
    return 'yes' if any([sent._.contains_pathogen and sent._.contains_disease for sent in doc.sents]) else 'no'