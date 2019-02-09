import gzip
import os
from pathlib import Path

import networkx as nx

import node2vec as n2v
from tqdm import tqdm

"""
 scratch file to test memory requirements
 TODO deleteme 
"""


def tqdm_pc(*args, mininterval=0.1, maxinterval=10, **kwargs):
    if "PYCHARM_HOSTED" in os.environ:
        mininterval = max(mininterval, 5)
        maxinterval = mininterval

    for x in tqdm(*args, **kwargs, mininterval=mininterval, maxinterval=maxinterval):
        yield x


def get_alias_edge(G, p, q, src, dst):

    unnormalized_probs = []
    for dst_nbr in sorted(G.neighbors(dst)):
        if dst_nbr == src:
            unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)  # return probability
        elif G.has_edge(dst_nbr, src):
            unnormalized_probs.append(G[dst][dst_nbr]['weight'])    # distance of 1
        else:
            unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)  # distance of > 1
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

    return n2v.alias_setup(normalized_probs)


def preprocess_node_transition_probs(G):

    alias_nodes = {}
    for node in tqdm_pc(G.nodes(), total=G.number_of_nodes(), desc='nodes'):
        unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        alias_nodes[node] = n2v.alias_setup(normalized_probs)

    return alias_nodes


def preprocess_edge_transition_probs(G):

    alias_edges = {}

    for edge in tqdm_pc(G.edges(), total=G.number_of_edges(), desc='edges'):
        alias_edges[edge] = get_alias_edge(G, .5, .5, edge[0], edge[1])

    return alias_edges


def node2vec_walk(G, p, q, walk_length, start_node):
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = sorted(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                # just starting the walk, take a random step from the node
                node = walk[0]
                unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
                norm_const = sum(unnormalized_probs)
                normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
                J, q = n2v.alias_setup(normalized_probs)

                walk.append(cur_nbrs[n2v.alias_draw(J,q)])
            else:
                # otherwise incorporate the search bias into the walk
                prev = walk[-2]
                J, q = get_alias_edge(G, .5, .5, prev, cur)

                next = cur_nbrs[n2v.alias_draw(J, q)]
                walk.append(next)
        else:
            break

    return walk


def walk_all_nodes(G, p, q, n=None):
    i = 0
    for start in tqdm_pc(G.nodes()):
        node2vec_walk(G, p, q, 80, start)
        i += 1
        if n and i == n:
            break


if __name__ == '__main__':

    def read_lines():
        with gzip.open(str(Path.home() / 'work/ner_light/edges.txt.gz'), 'r') as f:
            for line in tqdm_pc(f, total=48256298):
                yield line.decode('utf-8').strip()

    G = nx.parse_edgelist(read_lines(), nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
