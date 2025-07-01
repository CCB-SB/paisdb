# MODIFIED CODE VERSION FROM MASTER THESIS OF MARKUS DILLMANN
##################################################
# IMPORTS
##################################################
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

from tqdm import tqdm
from pprint import pprint
from utils.utils import get_substr_between2chars, process_string, fuzzy_search
import nltk
nltk.download('stopwords')



class Ontology:

    def __init__(self, location, name="Ontology"):
        import pronto
        # Directly downloand from OBO online library: https://obofoundry.org/ontology/doid.html
        self.ont = pronto.Ontology.from_obo_library(location)

        # Metadata information
        self.name = name
        self.version = self.ont.metadata.data_version
        self.url = self.ont.path

        self.relation_dict = dict()
        self.build_graph_doid()
        
		# Not implemented
		# self.tree = dict()
		# self.diseases = list()

    def build_graph_doid(self):
        """
        parsed the obo file into graph
        """
        for term in self.ont.terms():
            
            # Replace obsolete term by new one
            if term.obsolete:
                if term.replaced_by:
                    term = term.replaced_by.pop()
                else:
                    # print(f"no replacement for {term.id}")
                    continue
            
            # preprocess string
            # term.name = process_string(term.name)


            self.relation_dict[term.name] = {
                'id': term.id, 'name': term.name , 'definition': str(term.definition),
                "superclasses": {}, "subclasses": {},
                "has_exact_synonym": set(), "has_related_synonym": set(),
                'xref': set(term.xrefs) if len(term.xrefs) > 0 else set()}
            
            # parse relationships for each term i.e.: "is_a"
            for item in list(term.superclasses()):
                if item.name and item.id:
                    self.relation_dict[term.name]['superclasses'].setdefault("name", []).append(item.name.lower())
                    self.relation_dict[term.name]['superclasses'].setdefault("id", []).append(item.id)
            for item in list(term.subclasses()):
                if item.name and item.id:
                    self.relation_dict[term.name]['subclasses'].setdefault("name", []).append(item.name.lower())
                    self.relation_dict[term.name]['subclasses'].setdefault("id", []).append(item.id)

            # parse synonyms, distinguish between EXACT and NOT EXACT (RELATED, BROAD or NARROW)
            if len(term.synonyms) > 0:
                for elem in term.synonyms:
                    synonym_type = ""
                    if elem.scope == "EXACT":
                        synonym_type = "has_exact_synonym"
                    else:
                        # synonym type is RELATED, BROAD or NARROW
                        synonym_type = "has_related_synonym" 
                    self.relation_dict[term.name][synonym_type].add(elem.description.lower())
            

    def get_matching_node(self, query, scorer, threshold=90.1, string_process = True):
        from rapidfuzz import fuzz
        """
        :param query: search disease in ont
        :param threshold: minimal similarity for the matches
        :return: tupel, where first is the similarity score and second pos. is the dict entry
        """
        if string_process:
            query= process_string(query)
            # query_pros = process_string(query)
            # print(f"{query} -> {query_pros}")
            # query = query_pros
        
        if scorer == fuzz.WRatio:
            # Exact matches
            if query in self.relation_dict:
                best_match = {'score': 100, 'ontology': self.name,
                            'match': query, 'method': 'exact', 
                            'group': 'term', 'term': query, 'term_id': self.relation_dict[query]['id']}
                return best_match
        
        # Fuzzy Terms
        best_match = fuzzy_search(query = query, choice_list = list(self.relation_dict.keys()), 
                                    threshold = threshold, scorer = scorer)
        if best_match:
                best_match['ontology'] = self.name
                best_match['group'] = "term"
                best_match['term'] = best_match['match']
                best_match['term_id'] = self.relation_dict[best_match['term']]["id"]
                return best_match
        
        # Fuzzy Exact synonyms
        exact_syns = set()
        for key, value in self.relation_dict.items():
            exact_syns.update(value['has_exact_synonym'])
        
        
        best_match = fuzzy_search(query = query, choice_list = list(exact_syns), 
                                  threshold = threshold, scorer = scorer)
        if best_match:
                best_match['ontology'] = self.name
                best_match['group'] = "exact_synonym"
                # Obtain term associated
                terms = [(value["name"], value['id']) for key, value in self.relation_dict.items() for item in value['has_exact_synonym'] if item == best_match['match']]
                if len(terms) > 1:
                        print("Watch out! More than one term for the same exact synonym")
                        print(f"Terms for {best_match['match']}:")
                        print(terms)
                        print("Ambigous term, skipping...")
                        pass
                else:
                    best_match['term'] = terms[0][0]
                    best_match['term_id'] = terms[0][1]
                    return best_match
        
        # Fuzzy Related synonyms
        related_syns = set()
        for key, value in self.relation_dict.items():
            related_syns.update(value['has_related_synonym'])
        
        best_match = fuzzy_search(query = query, choice_list = list(related_syns),
                                  threshold = threshold, scorer = scorer)
        if best_match:
                best_match['ontology'] = self.name
                best_match['group'] = "related_synonym"
                                    # Obtain term associated
                terms = [(value["name"], value['id']) for key, value in self.relation_dict.items() for item in value['has_related_synonym'] if item == best_match['match']]
                if len(terms) > 1:
                        print("Watch out! More than one term for the same related synonym. Taking only the first one...")
                        print(f"Terms for {best_match['match']}")
                        print(terms)
                        print("Ambigous term, skipping...")
                        pass
                else:
                    best_match['term'] = terms[0][0]
                    best_match['term_id'] = terms[0][1]
                    return best_match
                

	# def eval_node(self, child):
		
	# 	# Eval if parent
	# 	parent = child.superclasses(with_self=False, distance=1).to_set()
		
	# 	if parent:
	# 		parent = parent.pop()
	# 		if parent.name in child.name:
	# 			return(self.eval_node(child = parent))
		
	# 	# Check if UMLS CUI
	# 	uid_umls = [ref for ref in child.xrefs if 'UMLS' in ref.id]
	# 	if uid_umls:
	# 		# Check if already in list
	# 		if child.name not in self.diseases:
	# 			self.diseases.append(child.name)
	# 			return
		
	# 	if parent:
	# 		return(self.eval_node(child = parent))
	

	# def get_leaf_diseases(self):
	# 	"""Generate a dict-like tree with the hierarchical organization
	# 	"""
	# 	leaves = [term for term in self.ont.terms() if term.is_leaf() if not term.obsolete]

	# 	# Discard diseases causes by infectious agent ('DOID:0050117') and genetic cause ('DOID:630')
	# 	infection_term = self.ont['DOID:0050117']
	# 	genetic_term = self.ont['DOID:630']
	# 	leaves = [leaf for leaf in leaves if infection_term not in leaf.superclasses(
	# 				).to_set() if genetic_term not in leaf.superclasses().to_set()]

	# 	for leaf in tqdm(leaves):
	# 		try:
	# 			self.eval_node(child=leaf)
	# 		except Exception as e:
	# 			print(leaf)
	# 			raise e
               
class Disease_ontology(Ontology):
    def __init__(self, location, name="Disease Ontology"):
        Ontology.__init__(self, location, name)

    def build_graph_doid(self):
        super().build_graph_doid()

        # identify key patterns in the term description
        for key, value in self.relation_dict.items():
            patterns = ['has_material_basis_in', 'has_symptom', 'derives_from', 'related_to']
            for pattern in patterns:
                if pattern in value['definition']:
                    self.relation_dict[key].setdefault(
                        pattern, 
                        get_substr_between2chars(
                            ch1 = pattern, ch2 = '[\.,]', s = value['definition'].lower()) 
                        ) 

if __name__ == '__main__':
	import sys
	from os.path  import abspath
	sys.path.insert(0, abspath("scripts"))
	from data_collection.parse_ontology import Ontology
	DO = Ontology(location="doid.obo")
	DO.get_leaf_diseases()

	len(DO.diseases)