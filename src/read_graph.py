'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import bz2
import gzip
import json
import random
from pathlib import Path

from gensim.models.word2vec import LineSentence
from tqdm import tqdm

import numpy as np
import networkx as nx
import os
from gensim.models import Word2Vec
from networkx.utils import open_file

from node2vec import InMemorySampler, SqliteSampler, GraphWalker, tqdm_pc, ConcurrentInMemorySampler


def arg_parser():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='../graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser


def read_graph(args):
    '''
    Reads the input network in networkx.
    '''

    @open_file(0, mode='rt')
    def parse_graph(f, args):
        if args.weighted:
            G = nx.parse_edgelist(tqdm_pc(f), nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        else:
            G = nx.parse_edgelist(tqdm_pc(f), nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

        if not args.directed:
            G = G.to_undirected()

        return G

    return parse_graph(args.input, args)




if __name__ == "__main__":
    G = read_graph(arg_parser().parse_args())
