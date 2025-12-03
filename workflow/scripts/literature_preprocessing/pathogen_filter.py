##################################################
# IMPORTS
##################################################

import networkx as nx
import numpy as np
import pandas as pd


##################################################
# FUNCTIONS
##################################################
def extract_lineage(row):
    lineage = [#(row['superkingdom'], row['superkingdom_id']),
               (row['phylum'], row['phylum_id']),
               (row['class'], row['class_id']),
               (row['order'], row['order_id']),
               (row['family'], row['family_id']),
               (row['genus'], row['genus_id']),
               (row['species'], row['species_id'])
               ]
    return lineage


def create_graph_from_csv(pathogen_csv):
    edges = []
    nodes = {'pathogen': -1}

    # fictional root node

    # extract lineage
    for index, row in pathogen_csv.iterrows():
        lineage = extract_lineage(row)
        parent_shift = 0
        for pos, (field, field_id) in enumerate(lineage):
            if field is None:
                parent_shift +=1
                continue
            # create node
            nodes[field] = field_id
            # create edge
            if pos == 0:
                edges.append((-1, field_id, "is_a")) #, 'pathogen', field))
            else:
                parent, parent_id = lineage[pos - parent_shift - 1]
                edges.append((parent_id, field_id, "is_a"))#, parent, field))
                parent_shift = 0

    # Create edge DataFrame for loading into graph
    df_edges = pd.DataFrame(edges, columns=["source", "target", "is_a"])

    # Create network
    G = nx.from_pandas_edgelist(df_edges, source="source", target="target", edge_attr="is_a", create_using=nx.DiGraph())
    root = nodes['pathogen']

    return G, nodes, root

# adds for a pathogen the disease pmid mapping or extends it by the pmid if mapping already exists.
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
        children_union = {}
        for child in G.successors(node):
            child_mapping = update_mappings(G,child)
            for disease in child_mapping:
                if disease in children_union:
                    children_union[disease].update(child_mapping[disease])
                else:
                    children_union[disease] = set(child_mapping[disease])

        # for each disease of the node create mapping set without children
        for disease in node_mapping:
            if disease in children_union:
                node_mapping[disease] = node_mapping[disease] - children_union[disease]

        # the union of children and parent is send up
        for disease in node_mapping:
            if disease in children_union:
                children_union[disease].update(node_mapping[disease])
            else:
                children_union[disease] = set(node_mapping[disease])
        #print(f"Has Children {node}: {children_union}")
        return children_union
    else:
        # no children -> no pmid present in children
        #print(f"Childless {node}: {node_mapping}")
        return node_mapping

##################################################
# MAIN
##################################################
if __name__ == '__main__':

    pathogen_csv_path = "" # Add the path to the csv containing the lineage here
    batches = 0 # Total number of batches
    input_path = "" # Path to the input folder
    output_path = "" # Path to the output folder

    # load pathogen data
    pathogen_csv = pd.read_csv(pathogen_csv_path)
    pathogen_csv.replace({np.nan: None}, inplace=True)

    # create graph
    G, node_mapper, root = create_graph_from_csv(pathogen_csv)

    # add to each node a dict which maps pathogens to a set of pmids
    for node in G.nodes():
        G.nodes[node]['pmids'] = {}

    # for all batches all entries add them
    # I used the names, however I would recommend using the pathogen and disease identifier directly for stability
    for batch in range(0,batches):
        csv = pd.read_csv(input_path + f"/batch_{batch}.csv")
        for disease, pathogen, pmid in zip(list(csv['disease']), list(csv['pathogen']), list(csv ['pmid'])):
            node = node_mapper[pathogen]
            add_entry(G,node,disease,pmid) # if pathogen is the identifier use add_entry(G, pathogen, disease, pmid) directly.

    # update the graph
    update_mappings(G,root)

    # check all entries if pmid still at node disease else mark for deletion
    for batch in range(0,batches):
        csv = pd.read_csv(input_path + f"/batch_{batch}.csv")
        marked_for_removal = []
        for disease, pathogen, pmid in zip(list(csv['disease']), list(csv['pathogen']), list(csv['pmid'])):
            node = node_mapper[pathogen]
            if pmid not in G.nodes[node]['pmids'][disease]:
                marked_for_removal.append(True)
            else:
                marked_for_removal.append(False)
        csv['duplicate_pathogen']=marked_for_removal
        csv.to_csv(output_path + f'/batch_{batch}.csv', index=False)
