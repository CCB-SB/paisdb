import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

from utils.relation_extraction import *
from utils.concept_extraction import *
from utils.strain import *
from utils.cohort import *
from utils.pathogen_validation import *
from utils.post_acute import *


class Test_manager:
    '''
    The test manager registers all algorithms implemented and makes them callable 
    for performtest.py
    When adding new tester, it is important that the key and the snakemake param are identical.
    '''
    def __init__(self):
        self.tests = {
            # Relationship tester
            "existence_based_relationship" : existence_based_relationship,
            "proximity_based_relationship" : proximity_based_relationship,
            "sentence_based_relationship" : sentence_based_relationship,

            # Concept tester
            "pure_concept_extraction" : pure_concept_extraction,
            "pure_concept_nlp_extraction": pure_concept_nlp_extraction,
            "sentence_co_occurrence_concept_extraction" : sentence_co_occurrence_concept_extraction,
            
            # strain tester
            "pure_regex_strain" : pure_regex_strain,
            "sentence_co_occurrence_strain" : sentence_co_occurrence_strain,
            
            # cohort tester
            "list_based_cohort_size" : list_based_cohort_size,
            "semtypes_based_cohort_size" : semtypes_based_cohort_size,
            "list_and_semtype_based_cohort_size" : list_and_semtype_based_cohort_size,
            
            #pathogen validation tester
            "existence_based_pathogen_validation": existence_based_pathogen_validation,
            "proximity_based_pathogen_validation": proximity_based_pathogen_validation,
            "sentence_co_occurrence_pathogen_validation": sentence_co_occurrence_pathogen_validation,
            
            #post acute tester
            "list_based_post_acute" : list_based_post_acute,
            "list_based_co_occurrence" : list_based_co_occurrence,
        }

    def get_tester(self, tester_name):
        if tester_name not in self.tests:
            raise KeyError(f"The tester {tester_name} does not exist in the test manager")
        return self.tests[tester_name]