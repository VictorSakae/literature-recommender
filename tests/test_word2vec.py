"""Synthetic Word2Vec tests: no real corpus or provider required."""
from copy import deepcopy
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from gensim.models import KeyedVectors, Word2Vec
import networkx as nx
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.feature_extraction.text import TfidfVectorizer

from literature_recommender.evaluation import evaluate_ranking
from literature_recommender.experiment import prepare_reference_query
from literature_recommender.recommender import hybrid_scores, rank_scores, rank_candidate_list
from literature_recommender.similarity import title_similarity, citation_similarities
from literature_recommender.word2vec import (
    tokenize_abstract, prepare_word2vec_corpus, train_word2vec,
    document_vector, word2vec_abstract_similarities, WORD2VEC_PARAMETERS,
)


def papers(ids, texts):
    return pd.DataFrame(dict(paper_id=ids, abstract=texts, title=['graph model'] * len(ids)), dtype='string')


def vectors():
    wv = KeyedVectors(vector_size=2)
    wv.add_vectors(['a', 'b', 'c', 'negative'], np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]], dtype=np.float32))
    return wv


class Word2VecTests(unittest.TestCase):
    def setUp(self):
        self.wv = vectors()
        self.data = papers(list('QABCM'), ['a', 'a', 'b', 'c', pd.NA])
        self.corpus = prepare_word2vec_corpus(self.data)

    def test_tokenization_matches_tfidf_policy(self):
        text = 'The X-2 café methods _ naïve; running! A.B'
        self.assertEqual(tokenize_abstract(text), ['the', 'x', '2', 'café', 'methods', '_', 'naïve', 'running', 'a', 'b'])
        analyze = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b').build_analyzer()
        self.assertEqual(tokenize_abstract(text), analyze(text))
        with self.assertRaises(TypeError):
            tokenize_abstract(None)

    def test_absences_never_sentences_or_vectors(self):
        corpus = prepare_word2vec_corpus(papers(list('ABCDEFG'), ['a', pd.NA, None, '', ' ', 'noabstract', 'a']))
        self.assertEqual(list(corpus), [['a'], ['a']])
        self.assertEqual(set(corpus.texts), {'A', 'G'})
        for p in 'BCDEF':
            self.assertIsNone(corpus.vector(p, self.wv))

    def test_invalid_metadata(self):
        for data in (papers(['A', 'A'], ['a', None]), papers([''], ['a']),
                     papers(['A'], [None]), papers(['A'], ['---'])):
            with self.subTest(data=data), self.assertRaises(ValueError):
                prepare_word2vec_corpus(data)
        with self.assertRaises(TypeError):
            prepare_word2vec_corpus(pd.DataFrame({'paper_id': ['A'], 'abstract': [123]}))

    def test_mean_occurrences_no_normalization_and_oov(self):
        np.testing.assert_allclose(document_vector(['a', 'b', 'a', 'OOV'], self.wv), [2/3, 1/3])
        np.testing.assert_array_equal(document_vector(['c'], self.wv), [1., 1.])
        for tokens in ([], ['OOV'], ['a', 'negative']):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                document_vector(tokens, self.wv)

    def test_all_oov_is_not_imputed(self):
        corpus = prepare_word2vec_corpus(papers(['Q', 'O', 'M'], ['a', 'unknown', None]))
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            word2vec_abstract_similarities(corpus, self.wv, 'Q', ['O', 'M'])
        self.assertIsNone(corpus.vector('M', self.wv))

    def test_cosine_and_local_mean(self):
        scores = word2vec_abstract_similarities(self.corpus, self.wv, 'Q', list('ABCM'))
        self.assertEqual(scores['A'], 1.)
        self.assertEqual(scores['B'], 0.)
        self.assertAlmostEqual(scores['C'], 1/sqrt(2))
        self.assertAlmostEqual(scores['M'], (1 + 1/sqrt(2))/3)
        self.assertEqual(word2vec_abstract_similarities(self.corpus, self.wv, 'Q', ['B', 'M'])['M'], 0.)

    def test_unknown_missing_query_empty_and_all_missing(self):
        self.assertEqual(word2vec_abstract_similarities(self.corpus, self.wv, 'Q', []), {})
        for query, ids, error in [('M', [], ValueError), ('unknown', [], KeyError),
                                  ('Q', ['unknown'], KeyError), ('Q', ['M'], ValueError)]:
            with self.subTest(query=query, ids=ids), self.assertRaises(error):
                word2vec_abstract_similarities(self.corpus, self.wv, query, ids)

    def test_deterministic_training_global_vocab_and_round_trip(self):
        data = papers(['Z', 'A', 'B'], ['graph model ' * 6, 'graph network ' * 6, 'outside ' * 6])
        corpus = prepare_word2vec_corpus(data)
        first = train_word2vec(corpus)
        second = train_word2vec(prepare_word2vec_corpus(data.iloc[::-1]))
        self.assertEqual(set(first.wv.key_to_index), {'graph', 'model', 'network', 'outside'})
        self.assertEqual(first.vector_size, 100)
        self.assertEqual(first.corpus_count, 3)
        self.assertEqual(first.corpus_total_words, 30)
        np.testing.assert_allclose(first.wv.vectors, second.wv.vectors, rtol=1e-7, atol=1e-8)
        self.assertEqual(first.sg, 0)
        self.assertEqual(first.workers, 1)
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'word2vec.model')
            first.save(path, separately=[])
            restored = Word2Vec.load(path)
            np.testing.assert_array_equal(first.wv.vectors, restored.wv.vectors)

    def test_empty_vocabulary_training_error(self):
        with self.assertRaisesRegex(ValueError, 'min_count'):
            train_word2vec(self.corpus)

    def test_no_mutation_refit_or_aliasing(self):
        before = self.data.copy(deep=True)
        tokens = ['a', 'b', 'a']
        wv_before = self.wv.vectors.copy()
        texts_before = deepcopy(self.corpus.texts)
        with patch('literature_recommender.word2vec.train_word2vec', side_effect=AssertionError('refit')):
            word2vec_abstract_similarities(self.corpus, self.wv, 'Q', iter(['A', 'M', 'A']))
            document_vector(tokens, self.wv)[:] = 999
            self.corpus.vector('Q', self.wv)[:] = 999
        self.assertEqual(tokens, ['a', 'b', 'a'])
        self.assertEqual(self.corpus.texts, texts_before)
        assert_frame_equal(self.data, before)
        np.testing.assert_array_equal(self.wv.vectors, wv_before)

    def test_synthetic_hybrid_existing_apis(self):
        ids = ['Q'] + [f'C{i:02}' for i in range(21)]
        data = papers(ids, ['a', None] + ['a'] * 10 + ['b'] * 10)
        graph = nx.DiGraph([('Q', 'C00'), ('Q', 'C01'), ('Q', 'external')])
        graph.add_edges_from(('C01', p) for p in ids[3:])
        plan = prepare_reference_query(data, graph, 'Q')
        truth = plan['relevant_ids'].copy()
        graph_before = deepcopy(graph)
        corpus = prepare_word2vec_corpus(data)
        abstract = word2vec_abstract_similarities(corpus, self.wv, 'Q', sorted(plan['eligible_ids']))
        title = {p: title_similarity('graph model', 'graph model') for p in abstract}
        citation = citation_similarities(graph, 'Q', abstract)
        ranking = rank_scores(hybrid_scores(abstract, title, citation))
        control_vectors = {p: corpus.vector(p, self.wv) for p in plan['abstracts']}
        control = rank_candidate_list(data, graph, 'Q', plan['candidate_ids'], control_vectors)
        self.assertEqual([p for p, _ in ranking], [r['paper_id'] for r in control['ranking']])
        self.assertEqual(control['excluded'], plan['excluded'])
        for row in control['ranking']:
            self.assertEqual(row['title_similarity'], title[row['paper_id']])
            self.assertEqual(row['citation_similarity'], citation[row['paper_id']])
        self.assertEqual(evaluate_ranking([p for p, _ in ranking], truth),
                         {'hit@1': 1., 'hit@5': .2, 'hit@10': .1, 'recall@20': 1.})
        self.assertEqual(truth, plan['relevant_ids'])
        self.assertTrue(nx.utils.graphs_equal(graph, graph_before))


if __name__ == '__main__':
    unittest.main()
