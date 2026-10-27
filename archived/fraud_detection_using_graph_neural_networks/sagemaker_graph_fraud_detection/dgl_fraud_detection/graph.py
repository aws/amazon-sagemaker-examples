import os
import re
import dgl
import numpy as np

from data import *


def get_edgelists(edgelist_expression, directory):
    if "," in edgelist_expression:
        return edgelist_expression.split(",")
    files = os.listdir(directory)
    compiled_expression = re.compile(edgelist_expression)
    return [filename for filename in files if compiled_expression.match(filename)]

def construct_graph(training_dir, edges, nodes, target_node_type, heterogeneous=True):
    if heterogeneous:
        print("Getting relation graphs from the following edge lists : {} ".format(edges))
        edgelists, id_to_node = {}, {}
        for i, edge in enumerate(edges):
            edgelist, id_to_node, src, dst = parse_edgelist(os.path.join(training_dir, edge), id_to_node, header=True)
            if src == target_node_type:
                src = 'target'
            if dst == target_node_type:
                dst = 'target'
            edgelists[(src, 'relation{}'.format(i), dst)] = edgelist
            print("Read edges for relation{} from edgelist: {}".format(i, os.path.join(training_dir, edge)))

            # reverse edge list so that relation is undirected
            edgelists[(dst, 'reverse_relation{}'.format(i), src)] = [(b, a) for a, b in edgelist]

        # get features for target nodes
        features, new_nodes = get_features(id_to_node[target_node_type], os.path.join(training_dir, nodes))
        print("Read in features for target nodes")
        # handle target nodes that have features but don't have any connections
        # if new_nodes:
        #     edgelists[('target', 'relation'.format(i+1), 'none')] = [(node, 0) for node in new_nodes]
        #     edgelists[('none', 'reverse_relation{}'.format(i + 1), 'target')] = [(0, node) for node in new_nodes]

        # add self relation
        edgelists[('target', 'self_relation', 'target')] = [(t, t) for t in id_to_node[target_node_type].values()]

        g = dgl.heterograph(edgelists)
        print(
            "Constructed heterograph with the following metagraph structure: Node types {}, Edge types{}".format(
                g.ntypes, g.canonical_etypes))
        print("Number of nodes of type target : {}".format(g.number_of_nodes('target')))

        g.nodes['target'].data['features'] = features

        id_to_node = id_to_node[target_node_type]

    else:
        sources, sinks, features, id_to_node = read_edges(os.path.join(training_dir, edges[0]),
                                                          os.path.join(training_dir, nodes))

        # add self relation
        all_nodes = sorted(id_to_node.values())
        sources.extend(all_nodes)
        sinks.extend(all_nodes)

        g = dgl.graph((sources, sinks))

        if features:
            g.ndata['features'] = np.array(features).astype('float32')

        print('read graph from node list and edge list')

        features = g.ndata['features']

    return g, features, id_to_node
