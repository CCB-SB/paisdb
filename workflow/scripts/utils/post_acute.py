
from collections import defaultdict

def list_based_post_acute(text, config):
    nlp = config['nlp']
    post_symptoms = config['post_symptoms']
    post_adjectives = config['post_adjectives']

    linker = nlp.get_pipe("scispacy_linker")
    doc = nlp(text)

    symptoms = defaultdict(set)

    for entity in doc.ents:
        for umls_ent in entity._.kb_ents:
            match = linker.kb.cui_to_entity[umls_ent[0]]
            # the UMLS term has to be an exact match in the entity
            if match.canonical_name.lower() in entity.text.lower():
                # check for adjectives
                ent_dep_token = [entity.root] + list(entity.root.children)

                for dependent in ent_dep_token:
                    if dependent.text.lower() not in match.canonical_name.lower() and dependent.dep_ == 'amod':
                        symptoms[match.canonical_name.lower()].add(dependent.text.lower())

    direct_score = 0
    indirect_score = 0

    for symptom, adjectives in symptoms.items():
        if symptom in post_symptoms:
            direct_score += 1
            for adjective in adjectives:
                if adjective in post_adjectives:
                    direct_score += 0.5
        else:
            for adjective in adjectives:
                if adjective in post_adjectives:
                    indirect_score += 1

    if direct_score > 0 and indirect_score > 0:
        return 'post with uncommon symptoms'
    elif direct_score > 0:
        return 'post'
    else:
        return 'potential post'
    

from spacy.tokens import Span
def list_based_co_occurrence(text, config):
    nlp = config['nlp']
    post_symptoms = config['post_symptoms']
    post_adjectives = config['post_adjectives']
    pathogen = config['pathogen']
    disease = config['disease']

    # add rules to nlp to detect pathogen or disease in a sentence
    Span.set_extension("contains_pathogen", getter=lambda span: pathogen in span.text, force=True)
    Span.set_extension("contains_disease", getter=lambda span: disease in span.text, force=True)

    linker = nlp.get_pipe("scispacy_linker")
    doc = nlp(text)

    symptoms = defaultdict(set)

    for entity in doc.ents:
        # local detection of pathogen or disease needed
        if not entity.sent._.contains_pathogen and not entity.sent._.contains_disease:
            continue

        for umls_ent in entity._.kb_ents:
            match = linker.kb.cui_to_entity[umls_ent[0]]
            # the UMLS term has to be an exact match in the entity
            if match.canonical_name.lower() in entity.text.lower():
                # check for adjectives
                ent_dep_token = [entity.root] + list(entity.root.children)

                for dependent in ent_dep_token:
                    if dependent.text.lower() not in match.canonical_name.lower() and dependent.dep_ == 'amod':
                        symptoms[match.canonical_name.lower()].add(dependent.text.lower())

    direct_score = 0
    indirect_score = 0

    for symptom, adjectives in symptoms.items():
        if symptom in post_symptoms:
            direct_score += 1
            for adjective in adjectives:
                if adjective in post_adjectives:
                    direct_score += 0.5
        else:
            for adjective in adjectives:
                if adjective in post_adjectives:
                    indirect_score += 1

    if direct_score > 0 and indirect_score > 0:
        return 'post with uncommon symptoms'
    elif direct_score > 0:
        return 'post'
    else:
        return 'potential post'