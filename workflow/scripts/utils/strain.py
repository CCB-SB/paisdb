
def handle_groups(match):
    return [named_match for named_match in match.groupdict().values() if named_match]




import re
def pure_regex_strain(text, config):
    # Patterns
    common_culturs = r"(?P<common_culture>((?:ATCC|DSM|JCM|NRRL|NCIMB|NCTC|CBS|CIP|NBRC|KCTC|CCUG|LMG|IFO|MTCC|BCRC|LSPQ|CCT|IMI|TISTR|nc).{0,4}[0-9]{3,7}))"
    sero_types = r"(?P<sero_type>(non-){0,1}O-[0-9]{1,5})"
    strain_local_ids = r"(?P<strain_local_ids>(?<=strain) (([A-Z]{1,3}[0-9]{0,3})){1,4}|([A-Z]{1,3}[0-9]{1,3}) (?=strain))"

    # Final Pattern
    pattern = re.compile(f"{common_culturs}|{sero_types}|{strain_local_ids}")

    strains = set()
    for match in pattern.finditer(text):
        strains.update(handle_groups(match))
    
    return strains

from spacy.tokens import Span
def sentence_co_occurrence_strain(text, config):
    pathogens = config["pathogen"]
    nlp = config["nlp"]

    Span.set_extension("contains_pathogen", getter=lambda span: any(pathogen.lower() in span.text.lower() for pathogen in pathogens), force=True)

    doc = nlp(text)
    strains = set()

    # Patterns
    common_culturs = r"(?P<common_culture>((?:ATCC|DSM|JCM|NRRL|NCIMB|NCTC|CBS|CIP|NBRC|KCTC|CCUG|LMG|IFO|MTCC|BCRC|LSPQ|CCT|IMI|TISTR|nc).{0-3} [0-9]{3,7}))"
    sero_types = r"(?P<sero_type>(non-){0,1}O-[0-9]{1,5})"
    strain_local_ids = r"(?P<strain_local_ids>(?<=strain) (([A-Z]{1,3}[0-9]{0,3})){1,4}|([A-Z]{1,3}[0-9]{1,3}) (?=strain))"

    # Final Pattern
    pattern = re.compile(f"{common_culturs}|{sero_types}|{strain_local_ids}")

    for sent in doc.sents:
        if sent._.contains_pathogen:
            for match in pattern.finditer(sent.text):
                strains.update(handle_groups(match))

    return strains