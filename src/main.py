'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import json
import random
from pathlib import Path

import numpy as np
import networkx as nx
import os
from gensim.models import Word2Vec

import node2vec


def arg_parser():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='../graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='../emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--seed', '-s', type=int,
                        help='set random seed')
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
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [[str(node) for node in walk] for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)

    # old = Word2Vec.load(args.output)
    # print('all close:', np.allclose(old.wv.vectors, model.wv.vectors))
    model.save(args.output)

    path = Path(args.output + '.txt')

    desc = {
        'seed': args.seed,
        'input': args.input,
        'walk-length': args.walk_length,
        'num-walks': args.num_walks
    }

    path.write_text(json.dumps(desc, indent=2))

    return model


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if args.seed is not None:
        if not os.environ.get('PYTHONHASHSEED') != args.seed:
            raise ValueError('environment variable PYTHONHASHSEED must also be set to ' + str(args.seed) + ' for reproducible results')
        random.seed(args.seed)
        np.random.seed(args.seed)
        args.workers = 1
        print('using seed ', args.seed)
    nx_G = read_graph(args)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)

    return learn_embeddings(walks, args)

if __name__ == "__main__":
    _args = arg_parser().parse_args()
    _model = main(_args)
