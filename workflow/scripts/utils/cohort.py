
def list_based_cohort_size(text, config):
    from spacy.matcher import PhraseMatcher
    cohort_names = config["cohort_names"]
    nlp = config["nlp"]

    # add custom entity detection for cohort
    phrasematcher = PhraseMatcher(nlp.vocab)
    pattern = [nlp.make_doc(cohort_name) for cohort_name in cohort_names]
    phrasematcher.add("list_cohort", pattern)

    cohort_sizes = []

    doc = nlp(text)

    # list based approach
    for match in phrasematcher(doc, as_spans=True):
        for child in match.root.children:
            if child.dep_ == 'nummod' and child.is_digit:
                try:
                    cohort_sizes.append(int(child.text))
                except:
                    continue

    return set([max(cohort_sizes)]) if len(cohort_sizes) > 0 else set()


def semtypes_based_cohort_size(text, config):
    nlp = config["nlp"]
    config_semtypes = set(config['config_semtypes'])

    cohort_sizes = []

    doc = nlp(text)
    linker = nlp.get_pipe("scispacy_linker")

    # semtype approach
    for entity in doc.ents:
        # check if ent is part of accepted semtypes
        for umls_ent in entity._.kb_ents:
            match = linker.kb.cui_to_entity[umls_ent[0]]
            semtypes = set(match.types)
            intersection = config_semtypes.intersection(semtypes)

            if len(intersection) > 0:
                for child in entity.root.children:
                    if child.dep_ == 'nummod' and child.is_digit:
                        try:
                            cohort_sizes.append(int(child.text))
                        except:
                            continue

    return set([max(cohort_sizes)]) if len(cohort_sizes) > 0 else set()

def list_and_semtype_based_cohort_size(text, config):
    from spacy.matcher import PhraseMatcher
    cohort_names = config["cohort_names"]
    nlp = config["nlp"]
    config_semtypes = set(config['config_semtypes'])

    # add custom entity detection for cohort
    phrasematcher = PhraseMatcher(nlp.vocab)
    pattern = [nlp.make_doc(cohort_name) for cohort_name in cohort_names]
    phrasematcher.add("list_cohort", pattern)

    cohort_sizes = []

    doc = nlp(text)
    linker = nlp.get_pipe("scispacy_linker")

    # list based approach
    for match in phrasematcher(doc, as_spans=True):
        for child in match.root.children:
            if child.dep_ == 'nummod' and child.is_digit:
                try:
                    cohort_sizes.append(int(child.text))
                except:
                    continue

    # semtype approach
    for entity in doc.ents:
        # check if ent is part of accepted semtypes
        for umls_ent in entity._.kb_ents:
            match = linker.kb.cui_to_entity[umls_ent[0]]
            semtypes = set(match.types)
            intersection = config_semtypes.intersection(semtypes)

            if len(intersection) > 0:
                for child in entity.root.children:
                    if child.dep_ == 'nummod' and child.is_digit:
                        try:
                            cohort_sizes.append(int(child.text))
                        except:
                            continue

    return set([max(cohort_sizes)]) if len(cohort_sizes) > 0 else set()
