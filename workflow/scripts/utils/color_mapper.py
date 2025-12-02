'''
this is a map used for the coloration of plots
to ensure uniform colors for the approaches
add more colors if needed for new approaches
'''
class colormapper24:
    def __init__(self):
        self.color_map = {
            'Biomistral-7B' : 'xkcd:mahogany',
            'Mistral-Small-Instruct-2409': 'xkcd:light blue',
            'pure_nlp': 'xkcd:violet',
            'pure_umls': 'xkcd:red orange',
            'sentence_co_occurrence_both_synonyms': 'xkcd:red',
            'sentence_co_occurrence_both': 'xkcd:blue',
            'sentence_co_occurrence_either': 'xkcd:tan',
            'sentence_co_occurrence_either_synonyms': 'xkcd:yellow',
            'list_and_semtype_based': 'xkcd:aqua green',
            'list_based': 'xkcd:light brown',
            'semtypes_based': 'xkcd:yellow green',
            'existence_based': 'xkcd:jade green',
            'existence_based_synonyms': 'xkcd:brown',
            'proximity_100_based': 'xkcd:green',
            'proximity_100_based_symonyms': 'xkcd:black',
            'proximity_200_based': 'xkcd:peach',
            'proximity_200_based_symonyms': 'xkcd:golden yellow',
            'proximity_400_based': 'xkcd:sky blue',
            'proximity_400_based_symonyms': 'xkcd:light gray',
            'proximity_800_based': 'xkcd:pink',
            'proximity_800_based_symonyms': 'xkcd:yellow orange',
            'sentence_co_occurrence': 'xkcd:magenta',
            'sentence_co_occurrence_synonyms': 'xkcd:orange',
            'pure_regex': 'xkcd:gray',
        }

    def colors(self):
        return list(self.color_map.values)

    def map(self, approach):
        return self.color_map.get(approach)