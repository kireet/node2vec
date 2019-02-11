import tempfile
from unittest import TestCase

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

import main as n2v_main


class CorrectnessTest(TestCase):

    def test_basic(self):
        with tempfile.TemporaryDirectory() as tempdir:
            args = n2v_main.arg_parser().parse_args(['--seed', '6278', '--output', f'{tempdir}/karate.emb'])

            n2v_main.main(args)

            exp = Word2Vec.load('../emb/karate.emb')
            act = Word2Vec.load(args.output)

            np.testing.assert_array_equal(exp.wv.vectors, act.wv.vectors)

    def test_file(self):
        with tempfile.TemporaryDirectory() as tempdir, tempfile.NamedTemporaryFile() as walks_file:
            args = n2v_main.arg_parser().parse_args(['--seed', '6278', '--output', f'{tempdir}/karate.emb', '--walks-file', walks_file.name])

            n2v_main.main(args)

            exp = Word2Vec.load('../emb/karate.emb')
            act = Word2Vec.load(args.output)

            np.testing.assert_array_equal(exp.wv.vectors, act.wv.vectors)

    def test_concurrent(self):
        with tempfile.TemporaryDirectory() as tempdir:
            args = n2v_main.arg_parser().parse_args(['--seed', '6278', '--concurrent-sampler', '--output', f'{tempdir}/karate.emb'])

            n2v_main.main(args)

            exp = Word2Vec.load('../emb/karate.emb')
            act = Word2Vec.load(args.output)

            np.testing.assert_array_equal(exp.wv.vectors, act.wv.vectors)

    def test_db_sampler(self):
        with tempfile.TemporaryDirectory() as tempdir, tempfile.NamedTemporaryFile() as db:
            args = n2v_main.arg_parser().parse_args(['--seed', '6278', '--output', f'{tempdir}/karate.emb', '--sample-file', db.name])

            n2v_main.main(args)

            exp = Word2Vec.load('../emb/karate.emb')
            act = Word2Vec.load(args.output)

            np.testing.assert_array_equal(exp.wv.vectors, act.wv.vectors)

    def test_max_edges(self):
        G = nx.DiGraph()

        G.add_edge(0, 1, weight=10)
        G.add_edge(0, 2, weight=2)
        G.add_edge(0, 3, weight=3)
        G.add_edge(0, 4, weight=5)

        G.add_edge(1, 0, weight=10)
        G.add_edge(2, 0, weight=10)
        G.add_edge(3, 0, weight=10)
        G.add_edge(4, 0, weight=10)

        n2v_main.prune_graph(G, 3)
        self.assertEqual(3, len(G[0]))
        for i in range(1, 5):
            self.assertEqual(1, len(G[i]))
            self.assertEqual(10, G[i][0]['weight'])

        self.assertEqual(10, G[0][1]['weight'])
        self.assertEqual(3, G[0][3]['weight'])
        self.assertEqual(5, G[0][4]['weight'])
