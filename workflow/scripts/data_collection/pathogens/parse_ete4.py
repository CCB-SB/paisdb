##################################################
# IMPORTS
##################################################
from multiprocessing import Value
import sys
from os.path  import abspath
sys.path.insert(0, abspath("scripts"))

##################################################
# FUNCTIONS
##################################################

from ete4 import NCBITaxa

class NCBITaxa_mod(NCBITaxa):
    def __init__(self):
      super().__init__()
      
      # Metadata
      self.name = "ete4"
      # Scientific, common names, and synonyms
      self.db_terms = {}

      self.inhouse_synonyms = {
         'Homo sapiens': {"human", "girl", "boy", 
                       "children", "infant", "male",
                       "female", "patient", "people",
                       "person", "baby", "adult"},
         'Bos taurus': {"cow", "beef", "cattle",
                        "bovine", "calf", "heifer"},
         'Mus musculus': {"mouse", "mice"},
         'Gallus gallus': {"chicken","hen", "broiler", "poultry"},
         'Sus scrofa': {"piglet", "pork", "pig","porcine", "swine"}
       }

    def get_ncbi_names(self):
         """
         """
         import tarfile
         from os.path import expanduser, join, exists
         import pandas as pd
         import pickle

         home_dir = expanduser("~")
         tax_dir = join(home_dir, ".local/share/ete/" )
         tax_dump = join(tax_dir, "taxdump.tar.gz")
         names_dump = join(tax_dir, 'names.dmp')
         
         # Obatin scientific, common, and synynonym names from names.dmp
         if not exists(names_dump):
            file = tarfile.open(tax_dump)
            file.extract('names.dmp', tax_dir) 
            file.close()
         
         # Parse names.dmp
         records = []
         name_types = ['equivalent name', 'synonym', 'acronym', 
                       'genbank common name', 'scientific name',
                       'common name', 'genbank acronym']
         with open(names_dump, 'r') as fh:
            for line in fh:
                line = line.strip("\t|\n")
                
                # taxid, term, category
                fields = line.split("\t|\t")

                if fields[3] in name_types:
                  records.append(fields)
         
         df = pd.DataFrame(records, columns=['taxid', 'term', 'unknown', 'category'])
         df.drop(columns=['unknown'], inplace=True)
         df['taxid'] = df['taxid'].astype(int)
         
         # Adjust names
         df.replace({
             'genbank common name': 'common name',
             'genbank acronym': 'acronym',
             'equivalent name': 'synonym'}, inplace=True)
         df.replace({
             'common name': 'common',
             'synonym': 'syn',
             'scientific name': 'sci',
             'acronym': 'acro'}, inplace=True)

         # Dictionary terms
         dict_terms_pickle = join(tax_dir, 'dict_terms.pickle')
         if not exists(dict_terms_pickle):
            dict_terms = {}
            for index, row in df.iterrows():
               dict_terms.setdefault(row['taxid'], {}).setdefault(
                  row['category'], set()).add(row['term'])
            pickle.dump(dict_terms, open(dict_terms_pickle, 'wb'))
         else:
            dict_terms = pickle.load(open(dict_terms_pickle, 'rb'))
         
         # Create terms lists
         self.db_terms = {
            'common': {key:entry['common'] for key, entry in dict_terms.items() if 'common' in entry.keys()},
            'sci': {key:entry['sci'] for key, entry in dict_terms.items() if 'sci' in entry.keys()},
            'syn': {key:entry['syn'] for key, entry in dict_terms.items() if 'syn' in entry.keys()},
            'acro': {key:entry['acro'] for key, entry in dict_terms.items() if 'acro' in entry.keys()}
            }
            
   
    def add_inhouse_synonyms(self):
      for key, items in self.inhouse_synonyms.items():
         name2taxid = self.get_name_translator([key])
         if name2taxid:
            host_taxid = list(name2taxid.values())[0][0]
            self.db_terms['common'].setdefault(host_taxid, set()).update(items)
            print(f"{key} common names:\n\t{self.db_terms['common'][host_taxid]}")
            

   
    def get_matching_node(self, query, scorer, threshold=90.1):
      from utils.utils import process_string, fuzzy_search
      
      query = process_string(query)

      if scorer == 'exact':
         # Exact match
         name2taxid = self.get_name_translator([query])
         if name2taxid:
            host_taxid= list(name2taxid.values())[0][0]
            host_name = list(self.get_taxid_translator([host_taxid]).values())[0]
            best_match = {'score': 100, 'category': self.name,
                           'match': host_name, 'term': host_name, 'method': 'exact', 
                           'gold_level': None, 'ID': host_taxid}
            return best_match
         
         return None
      
      # Fuzzy matches with Common, scientific, synonym and acronym names
      for type_name in ['common', 'sci', 'syn', 'acro']:
         choice_list = [item for item_list in self.db_terms[type_name].values() for item in item_list]
         best_match = fuzzy_search(query, 
                                    choice_list=choice_list, 
                                    threshold=threshold, scorer=scorer)
         if best_match:
            try:
                host_taxid = [key for key, entry in self.db_terms[type_name].items() if best_match['match'] in entry][0]
            except IndexError as e:
               print([key for key, entry in self.db_terms[type_name].items() if best_match['match'] in entry])
               raise e
            host_name = list(self.get_taxid_translator([host_taxid]).values())[0]
            best_match = {'score': best_match['score'], 'category': self.name,
                           'match': best_match['match'], 'term': host_name, 
                           'method':  best_match['method'], 
                           'gold_level': None, 'ID': host_taxid}
            return best_match
    
    def get_taxa_from_specie_taxid(self, sp_taxid, taxid=True, append_ID = False, 
                               lineage_list = ['superkingdom', 'phylum', 'class', 'order',
        'family', 'genus', 'species']):
        """
        Takes a specie taxID and returns a list with all the taxonomy levels
        
        Arguments
        ----------
            sp_taxid: int or str
                Integer containing the specie taxID or str
            taxid: bool
                Wheter the sp_taxid is taxid (int) or str(description)
            append_ID: bool
                Wheter to append taxid to the final list
            lineage_list: list
                Clade levels to include
            
        
        Returns
        -------
        list
            lineage taxonomy (i.e., Kingdom, Phylum,..., Specie, Strain) 
        
        """
        import re
        # Define 8-level taxa + formating prefixes
        

        if not taxid:
            name2taxid = self.get_name_translator([sp_taxid])
            if name2taxid:
                sp_taxid = list(name2taxid.values())[0][0]
            else:
                match = re.search(r'sp.?$', sp_taxid)
                if match:
                    sp_taxid = sp_taxid.replace(match.group(), '').strip()
                    return(self.get_taxa_from_specie_taxid(sp_taxid, taxid=False, 
                                            append_ID = True))
                else:
                    raise ValueError(f"No taxid found for {sp_taxid}") 

        lineage = self.get_lineage(sp_taxid) # contains the original order of taxids
        names = self.get_taxid_translator(lineage)
        # Get only 8-lineage (taxids)
        ranks = self.get_rank(lineage)
        # invert the ranks dictionary
        inv_ranks = {v: k for k, v in ranks.items()}

        value = []
        value_id = []
        last_name = ""
        last_rank = ""

        for rank in lineage_list:
            if rank in inv_ranks:
                id = inv_ranks[rank]
                value.append(names[id])
                value_id.append(id)
                last_name = names[id]
                last_rank = rank
            else:
                value.append("")
                value_id.append("")

        # replace spaces by _
        value = list(map(lambda x: x.replace(" ", "_"), value))

        if append_ID:
            value.append(sp_taxid)
            value_id.append(sp_taxid)

        return value, value_id, last_rank, last_name