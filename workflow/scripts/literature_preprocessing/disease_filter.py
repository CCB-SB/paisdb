##################################################
# IMPORTS
##################################################

import networkx as nx
import pronto
import pandas as pd


##################################################
# FUNCTIONS
##################################################
'''
    takes an pronto DO ontology and transforms it into a networkx directed graph
'''
def create_graph_from_obo(ontology):
    edges = []
    nodes = {}
    # Extract is_a relationships
    for term in ontology.terms():
        if term.name is None:
            continue
        nodes[term.name] = term.id
        integrated = False
        for parent in term.superclasses(distance=1, with_self=False):
            if parent.name:
                edges.append((parent.id, term.id, "is_a"))#, parent.name, term.name))
                integrated = True

        if not integrated and term.id != 'DOID:4':
            edges.append(('DOID:4', term.id, "is_a"))

    # Create edge DataFrame for loading into graph
    df_edges = pd.DataFrame(edges, columns=["source", "target", "is_a"])

    # Create network
    G = nx.from_pandas_edgelist(df_edges, source="source", target="target", edge_attr="is_a", create_using=nx.DiGraph())
    root = 'DOID:4' # disease DOID

    return G, nodes, root

# adds for a disease the pathogen pmid mapping or extends it by the pmid if mapping already exists.
def add_entry(G, node, pathogen, pmid):
    mapping = G.nodes[node]['pmids']
    if pathogen in mapping:
        mapping[pathogen].add(pmid)
    else:
        mapping[pathogen] = {pmid}

# update function to remove pmids which are present in at least 1 child
def update_mappings(G,node):
    node_mapping = G.nodes[node]['pmids']
    if len(list(G.successors(node))) > 0:
        # create union of mappings from children
        # descend to the child notes use their result to determine node mapping
        children_union = {}
        for child in G.successors(node):
            child_mapping = update_mappings(G,child)
            for pathogen in child_mapping:
                if pathogen in children_union:
                    children_union[pathogen].update(child_mapping[pathogen])
                else:
                    children_union[pathogen] = set(child_mapping[pathogen])

        # for each pathogen of the node create mapping set without children
        for pathogen in node_mapping:
            if pathogen in children_union:
                node_mapping[pathogen] = node_mapping[pathogen] - children_union[pathogen]

        # the union of children and parent is send up
        for pathogen in node_mapping:
            if pathogen in children_union:
                children_union[pathogen].update(node_mapping[pathogen])
            else:
                children_union[pathogen] = set(node_mapping[pathogen])
        return children_union
    else:
        # no children -> no pmid present in children
        return G.nodes[node]['pmids']

##################################################
# MAIN
##################################################
'''
This is the non snakemake integrated version used during the thesis
It is designed for using multiple batches, so ensure the files are distinguished by a integer
'''
if __name__ == '__main__':

    path_to_ontology = "" # Add the path to the ontology here
    batches = 0 # Total number of batches
    input_path = "" # Path to the input folder
    output_path = "" # Path to the output folder

    # load obo file for hierarchy
    ontology = pronto.Ontology(path_to_ontology)

    # create graph
    G, node_mapper, root = create_graph_from_obo(ontology)

    # add to each node a dict which maps pathogens to a set of pmids
    for node in G.nodes():
        G.nodes[node]['pmids'] = {}

    # for all batches add all entries to the graph
    # I used the names, however I would recommend using the pathogen and disease identifier directly for stability
    for batch in range(0,batches):
        csv = pd.read_csv(input_path + f"/batch_{batch}.csv")
        for disease, pathogen, pmid in zip(list(csv['disease']), list(csv['pathogen']), list(csv ['pmid'])):
            node = node_mapper[disease]
            add_entry(G,node,pathogen,pmid) # if disease is the identifier use add_entry(G, node, pathogen, pmid) directly.

    # update the graph
    update_mappings(G,root)

    # check all entries if pmid still at node disease else mark for deletion
    for batch in range(0,batches):
        csv = pd.read_csv(input_path + f"/batch_{batch}.csv")
        marked_for_removal = []
        for disease, pathogen, pmid in zip(list(csv['disease']), list(csv['pathogen']), list(csv['pmid'])):
            node = node_mapper[disease]
            if pmid not in G.nodes[node]['pmids'][pathogen]:
                marked_for_removal.append(True)
            else:
                marked_for_removal.append(False)
        csv['duplicate_disease']=marked_for_removal
        csv.to_csv(output_path + f"/batch_{batch}.csv", index=False)
