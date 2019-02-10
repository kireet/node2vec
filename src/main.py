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

    parser.add_argument('--output', nargs='?', default='../emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--walks-file',
                        help='store the walks in a file instead of memory.')

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

    parser.add_argument('--sample-file', '-sdb', type=str,
                        help='alias sample db')

    parser.add_argument('--concurrent-sampler', action='store_true',
                        help='concurrent in mem sampler')

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


def learn_embeddings(walks, args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
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

    if args.concurrent_sampler:
        print('using concurrent sampler')
    elif args.sample_file:
        print('using sqlite sampler')
    else:
        print('using in memory')

    G = read_graph(args)

    print('graph constructed.')

    if args.concurrent_sampler:
        sampler = ConcurrentInMemorySampler(G, args.p, args.q, args.directed, args.workers)
    elif args.sample_file:
        sampler = SqliteSampler(G, args.p, args.q, args.directed, Path(args.sample_file))
    else:
        sampler = InMemorySampler(G, args.p, args.q, args.directed)
    W = GraphWalker(G, sampler)
    walks = []

    def list_collector(w): walks.append([str(n) for n in w])

    walks_file = None

    if args.walks_file:
        walks_file_path = Path(args.walks_file)
        Path(args.walks_file).parent.mkdir(parents=True, exist_ok=True)
        walks_file = walks_file_path.open('wb') if walks_file_path.suffix != 'gz' else gzip.open('wb')

    def file_collector(w):
        s = ' '.join(map(str, w)) + '\n'
        walks_file.write(s.encode('utf-8'))

    collector = file_collector if walks_file else list_collector
    W.simulate_walks(args.num_walks, args.walk_length, args.workers, collector)

    if walks_file:
        walks_file.close()

    walks_iter = LineSentence(str(walks_file_path)) if walks_file else walks
    return W, learn_embeddings(walks_iter, args)


if __name__ == "__main__":
    _args = arg_parser().parse_args()
    _W, _model = main(_args)
