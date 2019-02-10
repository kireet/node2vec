import functools
import io
import os
import random
import sqlite3
import queue

import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm


# Lot of sorting going on. This is b/c the alias sampling arrays work off positions, but networkx graphs work off
# unordered collections, even though nx.OrderedDiGraph uses OrderedDict, it still doesn't support getting the nth key.
# so to avoid sorting, we'd have to also store a position to node id array with the alias sampling data. TBD if it's worthwhile.


def tqdm_pc(*args, mininterval=0.1, maxinterval=10, **kwargs):
    if "PYCHARM_HOSTED" in os.environ:
        mininterval = max(mininterval, 5)
        maxinterval = mininterval

    for x in tqdm(*args, **kwargs, mininterval=mininterval, maxinterval=maxinterval):
        yield x


class SamplerABC(ABC):

    def __init__(self, G:nx.Graph, p, q, is_directed:bool):
        """
        :param G: the networkx graph
        :param p: the return parameter, lower values encourge backtracking
        :param q: the in-out parameter, lower values encourage outward exploration
        :param is_directed: is the graph directed
        """
        self.G = G
        self.p = p
        self.q = q
        self.is_directed = is_directed

    @abstractmethod
    def sample(self, cur, prev=None) -> int:
        pass


def preprocess_alias_edge(G, p, q, src, dst) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the alias edge setup lists for a given edge. This contains the random walk probabilities
    if the walk just traversed the edge from src to dst.
    :param G: the networkx graph
    :param p: the return parameter, lower values encourge backtracking
    :param q: the in-out parameter, lower values encourage outward exploration
    :param src: the source node
    :param dst: the dest node
    :return: the normalized walk probabilities
    """

    unnormalized_probs = []
    for dst_nbr in sorted(G.neighbors(dst)):
        if dst_nbr == src:
            proba = G[dst][dst_nbr]['weight'] / p  # return probability
        elif G.has_edge(dst_nbr, src):
            proba = G[dst][dst_nbr]['weight']           # distance of 1
        else:
            proba = G[dst][dst_nbr]['weight'] / q  # distance of > 1

        unnormalized_probs.append(proba)

    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

    return alias_setup(normalized_probs)


def preprocess_alias_node(G, node):
    unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
    return alias_setup(normalized_probs)


class InMemorySampler(SamplerABC):
    def __init__(self, G:nx.Graph, p:float, q:float, is_directed:bool):
        """
        :param G: the networkx graph
        :param is_directed: is the graph directed
        :param p: the return parameter, lower values encourge backtracking
        :param q: the in-out parameter, lower values encourage outward exploration
        """

        self.G = G
        self.p = p
        self.q = q
        self.alias_nodes = None  #node2aliasdata dictionary
        self.alias_edges = None  #edge2aliasdata dictionary
        self.is_directed = is_directed
        self.preprocess_transition_probs()

    def sample(self, cur, prev=None) -> int:
        if prev is None:
            J, q = self.alias_nodes[cur]
        else:
            J, q = self.alias_edges[(prev, cur)]

        return alias_draw(J, q)

    def preprocess_transition_probs(self) -> None:
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        p = self.p
        q = self.q
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            alias_nodes[node] = preprocess_alias_node(G, node)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = preprocess_alias_edge(G, p, q, edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = preprocess_alias_edge(G, p, q, edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = preprocess_alias_edge(G, p, q, edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


_GRAPH: nx.Graph = None
_SAMPLER: SamplerABC = None

def worker(p, q, in_q: mp.Queue, out_q: mp.Queue):
    global _GRAPH

    items = in_q.get()
    while items is not None:
        if isinstance(items[0], tuple):
            out_q.put((items, [preprocess_alias_edge(_GRAPH, p, q, item[0], item[1]) for item in items]))
        elif isinstance(items[0], int):
            out_q.put((items, [preprocess_alias_node(_GRAPH, item) for item in items]))
        else:
            raise ValueError(str(items))
        items = in_q.get()

    print(os.getpid(), 'exiting...')


def _node_worker(node):
    return node, preprocess_alias_node(_GRAPH, node)


def _edge_worker(p, q, edge):
    return edge, preprocess_alias_edge(_GRAPH, p, q, *edge)


class ConcurrentInMemorySampler(InMemorySampler):
    def __init__(self, G:nx.Graph, p:float, q:float, is_directed:bool, workers):
        """
        :param G: the networkx graph
        :param p: the return parameter, lower values encourge backtracking
        :param q: the in-out parameter, lower values encourage outward exploration
        :param is_directed: is the graph directed
        """
        super().__init__(G, p, q, is_directed)
        self.workers = workers

    def preprocess_transition_probs(self) -> None:
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        global _GRAPH

        G = self.G
        p = self.p
        q = self.q

        is_directed = self.is_directed

        alias_nodes = {}
        # worker_q = mp.Queue()
        # output_q = mp.Queue()
        #
        # batch = []
        # num_sent, num_received = 0, 0
        _GRAPH = G

        with mp.Pool(self.workers) as pool:
            for node, J_q in tqdm_pc(pool.imap_unordered(_node_worker, G.nodes(), chunksize=1000), total=G.number_of_nodes(), desc='conc. in mem nodes'):
                alias_nodes[node] = J_q

        print('nodes computed')
        alias_edges = {}

        with mp.Pool(self.workers) as pool:
            num_edges = G.number_of_edges()
            edge_worker = functools.partial(_edge_worker, p, q)
            for edge, J_q in tqdm_pc(pool.imap_unordered(edge_worker, G.edges(), chunksize=1000), total=num_edges, desc='conc. in mem edges'):
                alias_edges[edge] = J_q

            if not is_directed:
                rev_edges = ((e[1], e[0]) for e in G.edges())
                for edge, J_q in tqdm_pc(pool.imap_unordered(edge_worker, rev_edges, chunksize=1000), total=num_edges, desc='conc. in mem back edges'):
                    alias_edges[edge] = J_q

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        _GRAPH = None

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


class SqliteSampler(SamplerABC):
    def __init__(self, G:nx.Graph, p:float, q:float, is_directed:bool, db_file:Path):
        """
        :param G: the networkx graph
        :param p: the return parameter, lower values encourge backtracking
        :param q: the in-out parameter, lower values encourage outward exploration
        :param is_directed: is the graph directed
        """

        super().__init__(G, p, q, is_directed)

        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("array", convert_array)

        self.alias_nodes = None  # node2aliasdata dictionary
        self.alias_edges = None  # edge2aliasdata dictionary
        self.db_file = db_file

        self.conn = sqlite3.connect(str(self.db_file), detect_types=sqlite3.PARSE_DECLTYPES)
        data_exists = self.create_schema()

        if data_exists:
            print('using existing database')
            _q = self.conn.execute('select value from params where id = "q"').fetchone()[0]
            _p = self.conn.execute('select value from params where id = "p"').fetchone()[0]
            if not np.allclose(p, float(_p)):
                raise ValueError(f'{_p} != {p}')
            if not np.allclose(q, float(_q)):
                raise ValueError(f'{_q} != {q}')
        else:
            print('creating new database')
            self.preprocess_transition_probs()

    @functools.lru_cache(100_000)
    def retrieve_alias_data(self, cur, prev=None):
        if prev is None:
            J, q = self.conn.execute('select J, q from alias_nodes where id = ?', (cur,)).fetchone()
        else:
            J, q = self.conn.execute('select J, q from alias_edges where prev = ? and cur = ?', (prev, cur)).fetchone()
        return J, q

    def sample(self, cur, prev=None) -> int:
        return alias_draw(*self.retrieve_alias_data(cur, prev))

    def create_schema(self):
        params_sql = """ CREATE TABLE IF NOT EXISTS params(
                                            id string PRIMARY KEY,
                                            value string not null
                                        ); """
        alias_nodes_sql = """ CREATE TABLE IF NOT EXISTS alias_nodes(
                                            id integer PRIMARY KEY,
                                            J array not null,
                                            q array not null
                                        ); """

        alias_edges_sql = """CREATE TABLE IF NOT EXISTS alias_edges (
                                        prev integer integer,
                                        cur integer integer,
                                        J array not null,
                                        q array not null,
                                        PRIMARY KEY (prev, cur)
                                    );"""

        self.conn.execute(params_sql)
        self.conn.execute(alias_nodes_sql)
        self.conn.execute(alias_edges_sql)

        if self.conn.execute('select count(*) from params').fetchone()[0] > 0:
            return True

        self.conn.execute("insert into params(id, value) values (?,?)", ('p', str(self.p)))
        self.conn.execute("insert into params(id, value) values (?,?)", ('q', str(self.q)))
        return False

    def preprocess_transition_probs(self) -> None:
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''

        for node in tqdm_pc(self.G.nodes(), desc='sqlite nodes', total=self.G.number_of_nodes()):
            J, q = preprocess_alias_node(self.G, node)
            self.conn.execute("insert into alias_nodes (id, J, q) values (?,?,?)", (node, J, q))

        alias_edges = {}

        if self.is_directed:
            for edge in tqdm_pc(self.G.edges(), desc='sqlite edges', total=self.G.number_of_edges()):
                J, q = preprocess_alias_edge(self.G, self.p, self.q, edge[0], edge[1])
                self.conn.execute("insert into alias_nodes (prev, cur, J, q) values (?,?,?,?)", (edge[0], edge[1], J, q))
        else:
            for edge in tqdm_pc(self.G.edges(), desc='sqlite edges', total=self.G.number_of_edges()):
                J, q = preprocess_alias_edge(self.G, self.p, self.q, edge[0], edge[1])
                self.conn.execute("insert into alias_edges (prev, cur, J, q) values (?,?,?,?)", (edge[0], edge[1], J, q))
                J, q = preprocess_alias_edge(self.G, self.p, self.q, edge[1], edge[0])
                self.conn.execute("insert into alias_edges (prev, cur, J, q) values (?,?,?,?)", (edge[1], edge[0], J, q))

class GraphWalker:

    def __init__(self, G:nx.Graph, sampler:SamplerABC):
        """
        :param G: the networkx graph
        :param sampler: the walk sampler
        """
        self.G = G
        self.sampler = sampler

    @staticmethod
    def node2vec_walk(G, sampler, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        :param G: the graph
        :param sampler: the sampler
        :param walk_length: the walk length
        :param start_node: the start node
        :return: the walk
        """

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            try:
                cur_nbrs = sorted(G.neighbors(cur))
            except TypeError:
                raise ValueError()
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    # just starting the walk, take a random step from the node
                    walk.append(cur_nbrs[sampler.sample(cur)])
                else:
                    walk.append(cur_nbrs[sampler.sample(cur=cur, prev=walk[-2])])
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length, cpu_count, collector):
        """
        Repeatedly simulate random walks from each node.
        :param num_walks: number of walks per node
        :param walk_length: the length of each walk
        :param cpu_count: number of worker processes
        :param collector: function to collect the walks
        """
        shuffled_nodes = list(self.G.nodes())

        def _inline_processor(nodes):
            for node in nodes:
                yield GraphWalker.node2vec_walk(G=self.G, sampler=self.sampler, walk_length=walk_length, start_node=node)

        def _mp_processor(nodes):
            global _GRAPH
            global _SAMPLER
            _GRAPH = self.G
            _SAMPLER = self.sampler
            with mp.Pool(cpu_count) as pool:
                for walk in pool.imap_unordered(functools.partial(_walk_worker, walk_length), nodes, chunksize=100):
                    yield walk

        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(shuffled_nodes)  # i assume this is to randomize the dataset order?
            iterable = _inline_processor(shuffled_nodes) if cpu_count == 1 else _mp_processor(shuffled_nodes)
            for walk in tqdm_pc(iterable, desc=f'walk {walk_iter}', total=self.G.number_of_nodes()):
                collector(walk)


def _walk_worker(walk_length, node):
    return GraphWalker.node2vec_walk(G=_GRAPH, sampler=_SAMPLER, walk_length=walk_length, start_node=node)


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
    J = np.zeros(K, dtype=np.int32)

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