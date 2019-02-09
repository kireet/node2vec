import random

import numpy as np
from tqdm import tqdm


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        """
        :param nx_G: the networkx graph
        :param is_directed: is the graph directed
        :param p: the return parameter, lower values encourge backtracking
        :param q: the in-out parameter, lower values encourage outward exploration
        """
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.alias_nodes = None  #node2aliasdata dictionary
        self.alias_edges = None  #edge2aliasdata dictionary

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        :param walk_length: the walk length
        :param start_node: the start node
        :return: the walk
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    # just starting the walk, take a random step from the node
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    # otherwise incorporate the search bias into the walk
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        :param num_walks: number of walks per node
        :param walk_length: the length of each walk
        :return: the walks
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in tqdm(range(num_walks)):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)  # i assume this is to randomize the dataset order?
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge. This contains the random walk probabilities
        if the walk just traversed the edge from src to dst.
        :param src: the source node
        :param dst: the dest node
        :return: the normalized walk probabilities
        """

        G = self.G
        p = self.p
        q = self.q

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

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details

    :param probs: the probabilities (length N)
    :return: q an array of length N where the value of q[i] is the probability of choice i given i was drawn uniformly. the
             remaining probability is allocated to the choice stored in J[i]
    """

    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob  # convert the probability to a condition prob Pr(kk | kk was drawn uniformly)
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    alias sample a discrete probability distribution.
    :param J: see alias_setup
    :param q: see alias_setup
    :return: the choice
    """
    K = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(np.random.rand()*K))  # basically np.random.choice

    # now that we've chosen a bucket, check to see if we should pick the original item or the "remainder" item
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]