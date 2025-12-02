####################################################
## The algorithms for the target organ and symptoms
####################################################

# see the pure_umls for details
def pure_concept_extraction(text, config):
    # load config
    quickuml = config['quick_umls']
    config_semtypes = set(config['config_semtypes'])
    # filter using QuickUMLS to check if they are from Anatomy
    hits = set()

    matches = quickuml.match(text, best_match=True, ignore_syntax=True)
    for match in matches:
        for m in match:
            semtypes = set(m['semtypes'])
            intersection = config_semtypes.intersection(semtypes)
            if len(intersection) > 0:
                hits.add(m['term'])

    return hits

from spacy.tokens import Span
from scispacy.linking import EntityLinker

# see the pure_nlp for details
def pure_concept_nlp_extraction(text, config):
    #load config
    nlp = config['nlp']
    config_semtypes = set(config['config_semtypes'])

    hits = set()

    doc = nlp(text)
    linker = nlp.get_pipe("scispacy_linker")

    for entity in doc.ents:
        # check if ent is part of config_semtypes
        for umls_ent in entity._.kb_ents:
            match = linker.kb.cui_to_entity[umls_ent[0]]
            semtypes = set(match.types)
            intersection = config_semtypes.intersection(semtypes)
            # the UMLS term has to be an exact match in the entity
            # and part of the config_semtypes
            if len(intersection) > 0 and match.canonical_name.lower() in entity.text.lower():
                hits.add(match.canonical_name)
    
    return hits

# sentence co occurrence and sentence co occurrence synonyms
# loaded pathogens and diseases determine synonym usage
def sentence_co_occurrence_concept_extraction(text, config):
    #load config
    nlp = config['nlp']
    pathogens = config['pathogen']
    diseases = config['disease']
    mode = config['mode']
    config_semtypes = set(config['config_semtypes'])

    hits = set()

    # add rules to nlp to detect pathogen or disease in a sentence
    Span.set_extension("contains_pathogen", getter=lambda span: any(pathogen in span.text for pathogen in pathogens), force=True)
    Span.set_extension("contains_disease", getter=lambda span: any(disease in span.text for disease in diseases), force=True)

    doc = nlp(text)
    linker = nlp.get_pipe("scispacy_linker")

    for entity in doc.ents:
        # check if ent is part of anatomy
        for umls_ent in entity._.kb_ents:
            match = linker.kb.cui_to_entity[umls_ent[0]]
            semtypes = set(match.types)
            intersection = config_semtypes.intersection(semtypes)
            # the UMLS term has to be an exact match in the entity
            # and part of the anatomy_semtypes
            if len(intersection) > 0 and match.canonical_name.lower() in entity.text.lower():
                origin_sent = entity.sent
                if mode == 'or':
                    if origin_sent._.contains_pathogen or origin_sent._.contains_disease:
                        hits.add(match.canonical_name)
                elif mode == 'and':
                    if origin_sent._.contains_pathogen and origin_sent._.contains_disease:
                        hits.add(match.canonical_name)

    return hits