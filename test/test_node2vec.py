import tempfile
from unittest import TestCase

from gensim.models import Word2Vec

import main as n2v_main
import numpy as np


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