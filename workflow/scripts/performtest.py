##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
from os import makedirs
from os.path import dirname
sys.path.insert(0, abspath("scripts"))

import pandas as pd
import numpy as np
from rapidfuzz import fuzz

from tqdm import tqdm

from test_manager import Test_manager

##################################################
# FUNCTIONS
##################################################

# function classifying the results from an testers extraction to the reference
def true_positive_matches(extracted, reference, mode):
    # make reference in a set if reference is not None
    reference = set(ref.strip() for ref in reference.split(',')) if reference else set()
    true_positive = set()
    false_positive = set()
    false_negative = set()

    if mode == 'string':
        if type(extracted) == str:
            extracted = [extracted]
        extracted = set(ext.strip() for ext in extracted) if extracted else set()

        used_extractions = set()

        # calculate true positives by checking which extracted are in the reference
        for ref in reference:
            for ext in extracted:
                edits = 1
                smaller_string_length =  min(len(ref), len(ext))
                threshold = 100 - edits / max(smaller_string_length, edits) * 100
                if fuzz.partial_ratio(ref.lower(), ext.lower()) >= threshold:
                    true_positive.add(ref)
                    used_extractions.add(ext)
                    continue

        if len(true_positive) == len(extracted) == len(reference) == 0:
            true_positive.add('Empty set')

        false_positive = extracted.difference(used_extractions) # extractions which are not in the reference ~ extractions which did not find a match in the references
        false_negative = reference.difference(true_positive) # references which where not covered by an extraction

    elif mode == 'int':
        # check if integer in reference
        reference = set(int(ref) for ref in reference)
        true_positive = extracted.intersection(reference)
        if len(true_positive) == len(extracted) == len(reference) == 0:
            true_positive.add('Empty set')
        false_positive = extracted.difference(reference)
        false_negative = reference.difference(extracted)

    else:
        raise Exception(f"Invalid mode {mode}")

    return true_positive, false_positive, false_negative, reference

# function classifying the results from an testers extraction to the reference
# specialised to get the distance as an additional parameter in the relationship detection
def relationship_with_distance(reference, distance, output_index):
    # reset score collectors
    extractions = []
    true_positives = []
    false_positives = []
    false_negatives = []
    classification = []

    #set current distance
    data['distance'] = distance
    
    # iterate over test set and determine if TP, FP, TN, FN
    for text, pathogen, disease, ground_truth in tqdm(zip(reference['text'], reference[pathogen_column], reference[disease_column], reference[reference_row]), total=len(reference)):
        data['pathogen'] = pathogen.split('|')
        data['disease'] = disease.split('|')

        extracted = tester(text, data)

        extractions.append(extracted)

        if extracted == ground_truth and extracted == 'yes':
            classification.append('TP')
        elif extracted == ground_truth and extracted == 'no':
            classification.append('TN')
        elif extracted == 'yes' and ground_truth == 'no':
            classification.append('FP')
        else:
            classification.append('FN')
    
    reference['extracted'] = extractions
    reference['classification'] = classification
    reference = reference[[pathogen_column,disease_column,reference_row,'extracted','classification']].copy()
    reference.rename(columns={reference_row: 'reference'}, inplace=True)
    output_path = snakemake.output.csvs[output_index]
    reference.to_csv(str(output_path), index=False)

# function classifying the results from an testers extraction to the reference
# specialised to get the distance as an additional parameter in the information retrieval approaches having the proximity approach type
def extraction_with_distance(reference, distance, output_index, mode):
    # reset score collectors
    extractions = []
    true_positives = []
    false_positives = []
    false_negatives = []
    classification = []

    #set current distance
    data['distance'] = distance
    # iterate over test and calculate metrics
    for text, pathogen, disease, ground_truth in tqdm(zip(reference['text'], reference[pathogen_column], reference[disease_column], reference[reference_row]), total=len(reference)):
        data['pathogen'] = pathogen.split('|')
        data['disease'] = disease.split('|')

        extracted = tester(text, data)
        TP, FP, FN, ground_truth = true_positive_matches(extracted, ground_truth, mode)

        extractions.append(list(extracted))
        true_positives.append(len(TP))
        false_positives.append(len(FP))
        false_negatives.append(len(FN))
    
    reference['extracted'] = extractions
    reference['TP'] = true_positives
    reference['FP'] = false_positives
    reference['FN'] = false_negatives

    reference = reference[[pathogen_column,disease_column,reference_row,'pmc','pmid','extracted','TP','FP', 'FN']].copy()
    reference.rename(columns={reference_row: 'reference'}, inplace=True)
    output_path = snakemake.output.csvs[output_index]
    reference.to_csv(str(output_path), index=False)

##################################################
# MAIN
##################################################
if __name__ == '__main__':
    reference = pd.read_csv(str(snakemake.input.csv))
    reference.replace({np.nan:None},inplace=True)

    test_parameter = snakemake.params

    reference_row = test_parameter['reference_row']


    # stores the meta data dependent on the parameter of snakemake.params
    data = {}
    if hasattr(test_parameter, 'nlp'):
        import warnings
        import spacy
        import scispacy
        from scispacy.linking import EntityLinker
        from scispacy.abbreviation import AbbreviationDetector

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nlp = spacy.load("en_core_sci_md")
            nlp.add_pipe("abbreviation_detector")
            nlp.add_pipe("scispacy_linker",
                     config={"resolve_abbreviations": True, "linker_name": "umls", "filter_for_definitions": False, "k": 5, "threshold": 0.95},
                     after="abbreviation_detector")

        data['nlp'] = nlp

    if hasattr(test_parameter, 'quick_umls'):
        from quickumls import QuickUMLS
        quickuml_matcher = QuickUMLS(test_parameter['quick_umls'],
            threshold=0.95, similarity_name='jaccard', window=5)
        data['quick_umls'] = quickuml_matcher

    if hasattr(test_parameter, 'cohort_names'):
        with open(test_parameter['cohort_names']) as f:
            cohort_names = [name.strip() for name in f.read().splitlines()]
            data['cohort_names'] = cohort_names

    if hasattr(test_parameter, 'config_semtypes'):
        with open(test_parameter['config_semtypes']) as f:
            semtypes = f.read().splitlines()
        data['config_semtypes'] = semtypes

    if hasattr(test_parameter, 'methods'):
        with open(test_parameter['methods']) as f:
            methods_names = [name.strip() for name in f.read().splitlines() if not name.startswith('#')]
        data['methods'] = methods_names

    if hasattr(test_parameter, 'distance'):
        distances = [int(distance) for distance in test_parameter['distance'].split(',')]

    if hasattr(test_parameter, 'mode'):
        data['mode'] = test_parameter['mode']

    if hasattr(test_parameter, 'post_symptoms'):
        with open(test_parameter['post_symptoms']) as f:
            post_symptoms = f.read().splitlines()
        data['post_symptoms'] = post_symptoms

    if hasattr(test_parameter, 'post_adjectives'):
        with open(test_parameter['post_adjectives']) as f:
            post_adjectives = f.read().splitlines()
        data['post_adjectives'] = post_adjectives

    pathogen_column = 'pathogen_term'
    disease_column = 'disease_term'
    if hasattr(test_parameter, 'use_synonyms'):
        pathogen_column = 'pathogen_synonyms'
        disease_column = 'disease_synonyms'

    # standard mode for evaluation is string
    # rules can change mode
    mode = 'string'
    if hasattr(test_parameter, 'result_type'):
        mode = test_parameter['result_type']
    
    # Scores:
    extractions = []
    true_positives = []
    false_positives = []
    false_negatives = []

    #special scores
    classification = []

    # llm testing
    if hasattr(test_parameter, 'llm'):
        # GPU based testing
        from utils.LLM_based_extraction_zero_shot import open_source_query
        answers = open_source_query(test_parameter['llm_model'], reference, snakemake.param['function'])

        # additional tracing invalid answers
        invalid_answers = 0

        for answer, reference in zip(answers, reference[reference_row]):
            if not answer:
                invalid_answers += 1
                answers = []

            # calculate metrics
            TP, FP, FN, ground_truth = true_positive_matches(extracted, ground_truth, mode)

            extractions.append(list(extracted))
            true_positives.append(len(TP))
            false_positives.append(len(FP))
            false_negatives.append(len(FN))

    # relationship detection testing
    elif hasattr(test_parameter, 'Relationship'): # different test set, different evaluation
        # Selecting the tester to use
        test_manager = Test_manager()
        tester = test_manager.get_tester(test_parameter['function'])

        if hasattr(test_parameter, 'distance'):
            for output_index, distance in enumerate(distances):
                relationship_with_distance(reference, distance, output_index)
            exit(0)
        
        # iterate over test set and determine if TP, FP, TN, FN
        for text, pathogen, disease, ground_truth in tqdm(zip(reference['text'], reference[pathogen_column], reference[disease_column], reference[reference_row]), total=len(reference)):
            data['pathogen'] = pathogen.split('|')
            data['disease'] = disease.split('|')

            extracted = tester(text, data)

            extractions.append(extracted)

            if extracted == ground_truth and extracted == 'yes':
                classification.append('TP')
            elif extracted == ground_truth and extracted == 'no':
                classification.append('TN')
            elif extracted == 'yes' and ground_truth == 'no':
                classification.append('FP')
            else:
                classification.append('FN')
        
        reference['extracted'] = extractions
        reference['classification'] = classification
        reference = reference[[pathogen_column,disease_column,reference_row,'extracted','classification']]
        reference.to_csv(str(snakemake.output.csv), index=False)
        # seperate exit as the result is different to the other tests
        exit(0)

    else:
        # Selecting the tester to use
        test_manager = Test_manager()
        tester = test_manager.get_tester(test_parameter['function'])

        if hasattr(test_parameter, 'distance'):
            for output_index, distance in enumerate(distances):
                extraction_with_distance(reference, distance, output_index, mode)
            exit(0)

        # iterate over test and calculate metrics
        for text, pathogen, disease, ground_truth in tqdm(zip(reference['text'], reference[pathogen_column], reference[disease_column], reference[reference_row]), total=len(reference)):
            data['pathogen'] = pathogen.split('|')
            data['disease'] = disease.split('|')

            extracted = tester(text, data)

            TP, FP, FN, ground_truth = true_positive_matches(extracted, ground_truth, mode)

            extractions.append(list(extracted))
            true_positives.append(len(TP))
            false_positives.append(len(FP))
            false_negatives.append(len(FN))


    reference['extracted'] = extractions
    reference['TP'] = true_positives
    reference['FP'] = false_positives
    reference['FN'] = false_negatives

    reference = reference[[pathogen_column,disease_column,reference_row,'pmc','pmid','extracted','TP','FP', 'FN']].copy()
    reference.rename(columns={reference_row: 'reference'}, inplace=True)

    reference.to_csv(str(snakemake.output.csv), index=False)