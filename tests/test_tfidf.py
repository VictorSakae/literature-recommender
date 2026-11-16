"""Corpus TF-IDF and controlled hybrid integration on synthetic inputs only."""

from copy import deepcopy
from math import log, sqrt
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.sparse import issparse

from literature_recommender.candidates import get_reference_list
from literature_recommender.evaluation import evaluate_ranking
from literature_recommender.recommender import hybrid_scores, rank_scores, rank_candidate_list
from literature_recommender.similarity import citation_similarities, title_similarity
from literature_recommender.tfidf import fit_tfidf, tfidf_abstract_similarities


def papers(ids, texts):
    return pd.DataFrame({'paper_id': ids, 'abstract': texts,
                         'title': ['graph model'] * len(ids)}, dtype='string')


class TfidfTests(unittest.TestCase):
    def setUp(self):
        self.papers = papers(['Q', 'A', 'B', 'C', 'M'],
                             ['common rare', 'common rare', 'other', 'common', pd.NA])
        self.corpus = fit_tfidf(self.papers)

    def scores(self, ids):
        return tfidf_abstract_similarities(self.corpus, 'Q', ids)

    def test_shared_vocabulary_idf_and_transform(self):
        v = self.corpus.vectorizer
        self.assertEqual(set(v.vocabulary_), {'common', 'rare', 'other'})
        for paper_id, text in self.papers[['paper_id', 'abstract']].itertuples(index=False, name=None):
            if pd.notna(text):
                transformed = v.transform([text])
                self.assertEqual((transformed != self.corpus.vector(paper_id)).nnz, 0)
        self.assertAlmostEqual(v.idf_[v.vocabulary_['common']], 1 + log(5 / 4))
        self.assertAlmostEqual(v.idf_[v.vocabulary_['rare']], 1 + log(5 / 3))
        self.assertGreater(v.idf_[v.vocabulary_['rare']], v.idf_[v.vocabulary_['common']])

    def test_identical_disjoint_and_numerical_partial(self):
        scores = self.scores(['A', 'B', 'C'])
        self.assertAlmostEqual(scores['A'], 1.)
        self.assertEqual(scores['B'], 0.)
        common, rare = 1 + log(5 / 4), 1 + log(5 / 3)
        expected = common / sqrt(common ** 2 + rare ** 2)
        self.assertAlmostEqual(scores['C'], expected, places=14)
        self.assertNotAlmostEqual(scores['C'], 1 / sqrt(2))
        self.assertTrue(all(type(value) is float for value in scores.values()))

    def test_sparse_mapping_independent_of_dataframe_index_and_order(self):
        shuffled = self.papers.iloc[::-1].copy()
        shuffled.index = [9] * len(shuffled)
        other = fit_tfidf(shuffled)
        self.assertTrue(issparse(other.matrix))
        self.assertEqual(other.paper_ids, ('A', 'B', 'C', 'Q'))
        self.assertEqual(other.row_by_paper_id, {'A': 0, 'B': 1, 'C': 2, 'Q': 3})
        self.assertEqual((other.matrix != self.corpus.matrix).nnz, 0)
        self.assertTrue(issparse(other.vector('A')))

    def test_absence_is_not_a_document_or_vector(self):
        data = papers(list('ABCDEF'), ['graph', pd.NA, 'noabstract', '', '  ', None])
        corpus = fit_tfidf(data)
        self.assertEqual(corpus.paper_ids, ('A',))
        self.assertEqual(corpus.matrix.shape, (1, 1))
        for paper_id in 'BCDEF':
            self.assertIsNone(corpus.vector(paper_id))
        self.assertEqual(corpus.vectorizer.vocabulary_, {'graph': 0})

    def test_imputation_uses_only_available_candidates_in_same_list(self):
        result = self.scores(['A', 'B', 'M'])
        self.assertAlmostEqual(result['M'], .5)
        self.assertAlmostEqual(self.scores(['A', 'M'])['M'], 1.)
        self.assertEqual(self.scores(['B', 'M'])['M'], 0.)
        with self.assertRaises(ValueError):
            self.scores(['M'])

    def test_empty_missing_query_unknown_ids_and_duplicates(self):
        self.assertEqual(self.scores([]), {})
        self.assertEqual(self.scores(iter(['A', 'M', 'A'])), self.scores(['A', 'M']))
        for ids in ([], ['A']):
            with self.assertRaises(ValueError):
                tfidf_abstract_similarities(self.corpus, 'M', ids)
        for operation in (lambda: self.scores(['absent']),
                          lambda: tfidf_abstract_similarities(self.corpus, 'absent', []),
                          lambda: self.corpus.vector('absent')):
            with self.assertRaises(KeyError):
                operation()

    def test_invalid_corpus(self):
        for data in (papers(['A', 'A'], ['text', pd.NA]),
                     papers([''], ['text']), papers(['A'], [pd.NA]),
                     papers(['A'], ['---'])):
            with self.subTest(data=data), self.assertRaises(ValueError):
                fit_tfidf(data)
        with self.assertRaises(TypeError):
            fit_tfidf(pd.DataFrame({'paper_id': ['A'], 'abstract': [123]}))

    def test_simple_text_policy(self):
        corpus = fit_tfidf(papers(['A'], ['The X-2 café methods _']))
        self.assertEqual(set(corpus.vectorizer.vocabulary_), {'the', 'x', '2', 'café', 'methods', '_'})
        self.assertEqual(corpus.vectorizer.transform(['cafe method']).nnz, 0)

    def test_no_refit_or_mutation(self):
        before = self.papers.copy(deep=True)
        corpus = fit_tfidf(self.papers)
        matrix = corpus.matrix.copy()
        vocab, idf = deepcopy(corpus.vectorizer.vocabulary_), corpus.vectorizer.idf_.copy()
        with patch.object(corpus.vectorizer, 'fit', side_effect=AssertionError('refit')), \
             patch.object(corpus.vectorizer, 'fit_transform', side_effect=AssertionError('refit')):
            for ids in (['A', 'M'], ['B', 'C']):
                tfidf_abstract_similarities(corpus, 'Q', ids)
        corpus.vector('A').data[:] = 99
        assert_frame_equal(before, self.papers)
        self.assertEqual((matrix != corpus.matrix).nnz, 0)
        self.assertEqual(vocab, corpus.vectorizer.vocabulary_)
        np.testing.assert_array_equal(idf, corpus.vectorizer.idf_)

    def test_corpus_includes_documents_outside_query_list(self):
        smaller = fit_tfidf(self.papers[self.papers.paper_id != 'B'])
        self.assertNotEqual(self.scores(['C'])['C'],
                            tfidf_abstract_similarities(smaller, 'Q', ['C'])['C'])

    def test_synthetic_reference_hybrid_changes_only_abstract_scores(self):
        ids = ['Q'] + [f'C{i:02}' for i in range(21)]
        data = papers(ids, ['graph model', pd.NA] + ['graph'] * 10 + ['other'] * 10)
        graph = nx.DiGraph([('Q', 'C00'), ('Q', 'C01'), ('Q', 'external')])
        graph.add_edges_from(('C01', p) for p in ids[3:])
        graph_before, data_before = deepcopy(graph), data.copy(deep=True)
        candidates = get_reference_list(graph, 'Q')
        eligible = sorted(candidates & set(data.paper_id))
        corpus = fit_tfidf(data)
        abstract = tfidf_abstract_similarities(corpus, 'Q', eligible)
        title = {p: title_similarity(data.iloc[0].title, data.set_index('paper_id').loc[p, 'title'])
                 for p in eligible}
        citation = citation_similarities(graph, 'Q', eligible)
        ranking = rank_scores(hybrid_scores(abstract, title, citation))
        # Existing neural path as control: title/citation, eligibility and truth identical.
        vectors = {p: None if pd.isna(text) else [1., 0.]
                   for p, text in data[['paper_id', 'abstract']].itertuples(index=False, name=None)}
        control = rank_candidate_list(data, graph, 'Q', candidates, vectors)
        self.assertEqual(eligible, control['eligible_ids'])
        self.assertEqual(control['excluded'], {'external': 'missing_metadata'})
        for row in control['ranking']:
            self.assertEqual(row['title_similarity'], title[row['paper_id']])
            self.assertEqual(row['citation_similarity'], citation[row['paper_id']])
        self.assertNotEqual(abstract['C20'], control['ranking'][-1]['abstract_similarity'])
        from literature_recommender.experiment import available_references
        truth = available_references(graph, 'Q', set(data.paper_id))
        self.assertEqual(truth, {'C00', 'C01'})
        self.assertEqual(evaluate_ranking([p for p, _ in ranking], truth),
                         {'hit@1': 1., 'hit@5': .2, 'hit@10': .1, 'recall@20': 1.})
        assert_frame_equal(data, data_before)
        self.assertTrue(nx.utils.graphs_equal(graph, graph_before))


if __name__ == '__main__':
    unittest.main()
