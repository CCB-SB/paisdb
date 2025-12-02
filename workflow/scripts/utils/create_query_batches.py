'''
The registry for which prompt should be used by a rule
def prompt using the functions below for reference and
add them in the batch creator
use the param function in the snakemake rule to refer to the prompt
'''
class batch_creator:
    def __init__(self):
        self.batch_functions = {
            "relationship_detection": create_query_batch_relationship_detection,
            "create_query_batch_cohort": create_query_batch_cohort,
            "create_query_batch_extract_symptoms": create_query_batch_extract_symptoms,
            "create_query_batch_extract_anatomy": create_query_batch_extract_anatomy,
            "create_query_batch_pathogen_validation": create_query_batch_pathogen_validation,
            "create_query_batch_strain": create_query_batch_strain,
        }

    def get_query_batch(self, query):
        if query not in self.batch_functions:
            raise KeyError(f'Query {query} is not defined')
        return self.batch_functions[query]

def create_query_batch_relationship_detection(df, tokenizer):
        queries = []
        for pathogen, disease, text in zip(
                df["pathogen"].to_list(),
                df["disease"].to_list(),
                df["text"].to_list(),
        ):
            query_relationship = f"""I seek assistance with a systematic review focused on the direct relationship between pathogens and diseases, specifically {disease}. I’ll provide the text of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

1. The text provides sufficient evidence of a direct relationship between the disease ({disease}) and the pathogen ({pathogen}).
2. The text investigates the Pathogen ({pathogen}) and reports evidence for the Disease ({disease}).
3. The text investigates the Disease ({disease}) and reports evidence for the Pathogen ({pathogen}).
4. The text states the association between the Pathogen ({pathogen}) and the Disease ({disease}), but does not focus on it.
5. The text present data or findings supporting this association.

Exclusion criteria:
1. The text do not provide sufficient evidence of a direct relationship between the disease ({disease}) and the pathogen ({pathogen}).

Please provide the assessment in the following dictionary format:
{{"relationship": 1, "unrelated": 0}} if there is a relationship, or {{"relationship": 0, "unrelated": 1}} if the study should be excluded.

Note: only one value can be 1 at a time.

Text: {text}

You are required to classify a journal article based solely on the given text. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the text.

Respond only in the dictionary format with no explanation.

Answer:"""
            queries.append(query_relationship)

        # If you plan to rely on device_map="auto",
        # you can keep these tokenized inputs on CPU – accelerate will handle distribution.
        queries_tokenized = tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            padding_side='left'

        )
        return queries, queries_tokenized


def create_query_batch_cohort(df, tokenizer):
    queries = []
    for pathogen, disease, text in zip(
            df["pathogen"].to_list(),
            df["disease"].to_list(),
            df["text"].to_list(),
    ):
        query_relationship = f"""I seek assistance with a systematic review focused on the extraction of the cohort size. I’ll provide the text of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

1. The text mentions an experiment or study.
2. Only the total amount of the cohort size should be mentioned.
3. If there are multiple cohorts their sum should be mentioned, check if one cohort is the total size, while the other are subcohorts!


Please provide the assessment in the following dictionary format:
{{"Cohort size": number}} if there is a cohort size, or {{"Cohort size": None}} if no cohort size could be found.

Text: {text}

You are required to classify a journal article based solely on the given text. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the text.

Respond only in the dictionary format with no explanation.

Answer:"""
        queries.append(query_relationship)

    # If you plan to rely on device_map="auto",
    # you can keep these tokenized inputs on CPU – accelerate will handle distribution.
    queries_tokenized = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side='left'

    )
    return queries, queries_tokenized


def create_query_batch_extract_symptoms(df, tokenizer):
    queries = []
    for pathogen, disease, text in zip(
            df["pathogen"].to_list(),
            df["disease"].to_list(),
            df["text"].to_list(),
    ):
        query_relationship = f"""I seek assistance with a systematic review focused on the direct extraction of symptoms related to pathogens and diseases, specifically {disease}. I’ll provide the text of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

1. The symptoms are mentioned in the context of the Pathogen ({pathogen}) and the Disease ({disease})
2. Categories include: Acquired Abnormality, Anatomical Abnormality, Cell or Molecular Dysfunction, Congenital Abnormality, Experimental Model of Disease, Finding, Injury or Poisoning, Mental or Behavioral Dysfunction, Sign or Symptom


Please provide the assessment in the following dictionary format:
{{"symptoms": "list of symptoms"}} if there are symptoms, or {{"symptoms": "[]"}} if no symptoms could be found.

Text: {text}

You are required to classify a journal article based solely on the given text. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the text.

Respond only in the dictionary format with no explanation.

Answer:"""
        queries.append(query_relationship)

    # If you plan to rely on device_map="auto",
    # you can keep these tokenized inputs on CPU – accelerate will handle distribution.
    queries_tokenized = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side='left'

    )
    return queries, queries_tokenized


def create_query_batch_extract_anatomy(df, tokenizer):
    queries = []
    for pathogen, disease, text in zip(
            df["pathogen"].to_list(),
            df["disease"].to_list(),
            df["text"].to_list(),
    ):
        query_relationship = f"""I seek assistance with a systematic review focused on the direct extraction of target organs / target cells related to pathogens and diseases, specifically {disease}. I’ll provide the text of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

1. The target organ / cells / body part are mentioned in the context of the Pathogen ({pathogen}) and the Disease ({disease})
2. For the Analysis target organ and target cells are all part of the anatomy
3. targets are neither diseases nor pathogens!!!
4. The dictionary only contains the answers to anatomy.

Please provide the assessment in the following dictionary format:
{{"anatomy": "list of targets"}} if there are targets, or {{"anatomy": "[]"}} if no targets could be found.

Text: {text}

You are required to classify a journal article based solely on the given text. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the text.

Respond only in the dictionary format with no explanation.

Answer:"""
        queries.append(query_relationship)

    # If you plan to rely on device_map="auto",
    # you can keep these tokenized inputs on CPU – accelerate will handle distribution.
    queries_tokenized = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side='left'

    )
    return queries, queries_tokenized


def create_query_batch_pathogen_validation(df, tokenizer):
    queries = []
    for pathogen, text in zip(
            df["pathogen"].to_list(),
            df["text"].to_list(),
    ):
        query_relationship = f"""I seek assistance with a systematic review focused on the direct extraction of validation methods used to identify pathogens, specifically {pathogen}. I’ll provide the text of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

1. The validation methods are mentioned in the context of the Pathogen ({pathogen}).

Examples of validation methods: 
- Enzyme immunoassay (EIA), enzyme-linked fluorescent assay (ELAF), enzyme-linked immunosorbent assay (ELISA), 
- polymerase chain reaction (PCR), reverse transcript PCR (RT-PCR), real-time PCR (qPCR), multiplexe PCR, nested PCR
- next generation sequencing (NGS)
- other metabolic techniques, spectroscopy technology, surface enhanced raman scattering, SERS, SPR, Mass spectrometry, flow cytometry, colorimetry

Please provide the assessment in the following dictionary format:
{{"pathogen_validation": "list of methods"}} if there are validation methods, or {{"pathogen_validation": "[]"}} if no validation methods could be found.

Text: {text}

You are required to classify a journal article based solely on the given text. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the text.

Respond only in the dictionary format with no explanation.

Answer:"""
        queries.append(query_relationship)

    # If you plan to rely on device_map="auto",
    # you can keep these tokenized inputs on CPU – accelerate will handle distribution.
    queries_tokenized = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side='left'

    )
    return queries, queries_tokenized


def create_query_batch_strain(df, tokenizer):
    queries = []
    for pathogen, text in zip(
            df["pathogen"].to_list(),
            df["text"].to_list(),
    ):
        query_relationship = f"""I seek assistance with a systematic review focused on the direct extraction of strain identifier used to identify pathogens species strain, specifically {pathogen}. I’ll provide the text of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

1. The strains are mentioned in the context of the Pathogen ({pathogen}).
2. Strains are identifications for a subset of a species with a common genomic region, specific to the subset. Common Identifications are <Culture Collection> <Number>, O-<number>, and <number and letters> strain.

Please provide the assessment in the following dictionary format:
{{"strains": "List of strains"}} if there are strain identifier, or {{"strains": "[]"}} if no strain identifier could be found.

Text: {text}

You are required to classify a journal article based solely on the given text. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the text.

Respond only in the dictionary format with no explanation.

Answer:"""
        queries.append(query_relationship)

    # If you plan to rely on device_map="auto",
    # you can keep these tokenized inputs on CPU – accelerate will handle distribution.
    queries_tokenized = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side='left'

    )
    return queries, queries_tokenized


def create_query_batch_post_acute(df, tokenizer):
    queries = []
    for pathogen, disease, text in zip(
            df["pathogen"].to_list(),
            df["disease"].to_list(),
            df["text"].to_list(),
    ):
        query_relationship = f"""I seek assistance with a systematic review focused on the post acute infection syndrome information between pathogens and diseases, specifically {disease}. I’ll provide the text of a particular journal article and would appreciate an assessment for its inclusion based on the following criteria:

    Post-acute infection syndrome are defined by persistend symptoms occuring after an infection with a pathogen.
    Focus on the pathogen {pathogen} and the disease {disease} and in their context post acute infections.

    Please provide the assessment in the following dictionary format:
    {{"post_acute": 1 }} if there is a post-acute syndrome evidence evidence, or {{"post_acute": 0 }} if there is no evidence of post-acute infection syndrome evidence.

    Note: only values are 1 or 0.

    Text: {text}

    You are required to classify a journal article based solely on the given text. Do not use any external knowledge or assumptions beyond the text provided. Your decision must be strictly based on the information within the text.

    Respond only in the dictionary format with no explanation.

    Answer:"""
        queries.append(query_relationship)

    # If you plan to rely on device_map="auto",
    # you can keep these tokenized inputs on CPU – accelerate will handle distribution.
    queries_tokenized = tokenizer(
        queries,
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side='left'

    )
    return queries, queries_tokenized