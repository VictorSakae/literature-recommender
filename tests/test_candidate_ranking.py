"""One-list integration with synthetic metadata, directed graph and embeddings."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import networkx as nx
import pandas as pd
from pandas.testing import assert_frame_equal

from literature_recommender.candidates import get_citation_list, get_reference_list
from literature_recommender.recommender import rank_candidate_list
from literature_recommender.similarity import abstract_similarities, citation_similarities


class CandidateRankingTests(unittest.TestCase):
    def setUp(self):
        self.papers = pd.DataFrame({
            'paper_id': ['Q', 'A', 'B', 'C'],
            'title': ['graph model', 'graph model', 'graph theory', 'other topic'],
            'abstract': ['query abstract', 'abstract A', 'abstract B', pd.NA],
        }, dtype='string')
        self.graph = nx.DiGraph([
            ('Q', 'R'), ('Q', 'S'), ('Q', 'T'),
            ('A', 'R'), ('B', 'R'), ('B', 'S'),
            ('C', 'R'), ('C', 'S'), ('C', 'T'),
            ('A', 'Q'), ('B', 'Q'), ('C', 'Q'), ('X', 'Q'),
        ])
        self.candidates = get_citation_list(self.graph, 'Q')
        self.embeddings = {'Q': [1., 0.], 'A': [1., 0.], 'B': [0., 1.], 'C': None}

    def rank(self, **kwargs):
        return rank_candidate_list(self.papers, self.graph, 'Q', self.candidates,
                                   self.embeddings, **kwargs)

    def test_manual_components_and_final_ranking(self):
        self.assertEqual(self.candidates, {'A', 'B', 'C', 'X'})
        result = self.rank()
        self.assertEqual(result['eligible_ids'], ['A', 'B', 'C'])
        self.assertEqual(result['excluded'], {'X': 'missing_metadata'})
        self.assertEqual([r['paper_id'] for r in result['ranking']], ['A', 'C', 'B'])
        # Shared-reference counts A=1, B=2, C=3 -> 0, .5, 1 after filtering X.
        # Abstracts: 1, 0, mean(1, 0)=.5. Titles: 1, .5, 0.
        expected = {'A': (1., 1., 0., .8), 'B': (0., .5, .5, .2), 'C': (.5, 0., 1., .5)}
        fields = ('abstract_similarity', 'title_similarity', 'citation_similarity', 'weighted_similarity')
        for row in result['ranking']:
            self.assertEqual(set(row), {'paper_id', *fields})
            for field, value in zip(fields, expected[row['paper_id']]):
                self.assertAlmostEqual(row[field], value, places=14)

    def test_exclusion_precedes_all_similarities(self):
        # If X entered citation normalization, A would score 1/3, not 0.
        # If X entered abstract scoring, its supplied vector would change C's mean.
        self.embeddings['X'] = [-1., 0.]
        with patch('literature_recommender.recommender.citation_similarities',
                   wraps=citation_similarities) as citations, \
             patch('literature_recommender.recommender.abstract_similarities',
                   wraps=abstract_similarities) as abstracts:
            result = self.rank()
        self.assertEqual(citations.call_args.args[2], ['A', 'B', 'C'])
        self.assertEqual(set(abstracts.call_args.args[1]), {'A', 'B', 'C'})
        self.assertEqual(result['ranking'][0]['citation_similarity'], 0.)
        self.assertEqual(result['ranking'][1]['abstract_similarity'], .5)

    def test_top_k_applies_after_all_scores(self):
        full = self.rank()
        for k in (1, 2, 20, None):
            with self.subTest(k=k):
                result = self.rank(top_k=k)
                self.assertEqual(result['ranking'], full['ranking'][:k])
                self.assertEqual(result['excluded'], full['excluded'])
                self.assertEqual(result['eligible_ids'], full['eligible_ids'])

    def test_custom_weights(self):
        result = self.rank(weights=(0, 0, 1))
        self.assertEqual([r['paper_id'] for r in result['ranking']], ['C', 'B', 'A'])

    def test_independent_lists_preserve_overlap(self):
        self.graph.add_edge('Q', 'A')
        references = get_reference_list(self.graph, 'Q')
        citations = get_citation_list(self.graph, 'Q')
        reference_result = rank_candidate_list(self.papers, self.graph, 'Q', references, self.embeddings)
        citation_result = rank_candidate_list(self.papers, self.graph, 'Q', citations, self.embeddings)
        self.assertEqual(reference_result['eligible_ids'], ['A'])
        self.assertEqual(citation_result['eligible_ids'], ['A', 'B', 'C'])
        self.assertEqual(reference_result['ranking'][0]['paper_id'], 'A')
        self.assertIn('A', [r['paper_id'] for r in citation_result['ranking']])

    def test_mean_and_normalization_are_local_to_list(self):
        self.candidates = {'A', 'C'}
        rows = {r['paper_id']: r for r in self.rank()['ranking']}
        self.assertEqual(rows['C']['abstract_similarity'], 1.)
        self.candidates = {'B', 'C'}
        rows = {r['paper_id']: r for r in self.rank()['ranking']}
        self.assertEqual(rows['C']['abstract_similarity'], 0.)
        self.assertEqual(rows['B']['citation_similarity'], 0.)

    def test_generator_duplicates_and_input_order(self):
        expected = self.rank()
        self.candidates = iter(['X', 'C', 'B', 'A', 'B'])
        self.assertEqual(self.rank(), expected)

    def test_empty_and_all_excluded(self):
        for candidates in (set(), {'X'}):
            self.candidates = candidates
            result = self.rank()
            self.assertEqual(result['ranking'], [])
            self.assertEqual(result['eligible_ids'], [])
            self.assertEqual(result['excluded'], {p: 'missing_metadata' for p in candidates})

    def test_missing_embedding_key_is_not_missing_abstract(self):
        for paper_id in ('Q', 'A', 'C'):
            with self.subTest(paper_id=paper_id):
                embeddings = self.embeddings.copy()
                del embeddings[paper_id]
                with self.assertRaises(KeyError) as error:
                    rank_candidate_list(self.papers, self.graph, 'Q', self.candidates, embeddings)
                self.assertEqual(error.exception.args, (paper_id,))

    def test_none_with_available_abstract_is_rejected(self):
        self.embeddings['A'] = None
        with self.assertRaisesRegex(ValueError, 'A.*exactly when'):
            self.rank()

    def test_vector_with_missing_abstract_is_rejected(self):
        self.embeddings['C'] = [1., 0.]
        with self.assertRaisesRegex(ValueError, 'C.*exactly when'):
            self.rank()

    def test_invalid_candidate_vectors_are_not_imputed_even_with_zero_weight(self):
        for invalid in ([0, 0], [float('nan'), 0], [1, 0, 0], 'bad'):
            self.embeddings['A'] = invalid
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, TypeError)):
                self.rank(weights=(0, 1, 0))

    def test_query_requires_valid_embedding_even_when_empty(self):
        for candidates in (self.candidates, set(), {'X'}):
            for invalid in (None, [0., 0.], [float('inf'), 0.]):
                self.candidates = candidates
                self.embeddings['Q'] = invalid
                with self.subTest(candidates=candidates, invalid=invalid), self.assertRaises((TypeError, ValueError)):
                    self.rank()

    def test_query_requires_metadata_and_available_abstract(self):
        self.papers = self.papers[self.papers.paper_id != 'Q']
        with self.assertRaises(KeyError):
            self.rank()
        self.setUp()
        self.papers.loc[self.papers.paper_id == 'Q', 'abstract'] = pd.NA
        with self.assertRaisesRegex(ValueError, 'Query.*abstract'):
            self.rank()

    def test_graph_lookup_errors_propagate(self):
        for paper_id in ('Q', 'A'):
            with self.subTest(paper_id=paper_id):
                graph = self.graph.copy()
                graph.remove_node(paper_id)
                with self.assertRaises(KeyError):
                    rank_candidate_list(self.papers, graph, 'Q', self.candidates, self.embeddings)

    def test_invalid_titles_are_not_excluded_or_zeroed(self):
        for paper_id in ('Q', 'A'):
            papers = self.papers.copy()
            papers.loc[papers.paper_id == paper_id, 'title'] = pd.NA
            with self.subTest(paper_id=paper_id), self.assertRaises(TypeError):
                rank_candidate_list(papers, self.graph, 'Q', self.candidates, self.embeddings)

    def test_duplicate_metadata_is_not_missing_metadata(self):
        for paper_id in ('Q', 'A'):
            papers = pd.concat([self.papers, self.papers[self.papers.paper_id == paper_id]])
            with self.subTest(paper_id=paper_id), self.assertRaisesRegex(ValueError, 'Duplicate article ID'):
                rank_candidate_list(papers, self.graph, 'Q', self.candidates, self.embeddings)

    def test_all_eligible_abstracts_missing_fails_without_mean(self):
        self.candidates = {'C', 'X'}
        with self.assertRaisesRegex(ValueError, 'No candidate abstract embeddings'):
            self.rank()

    def test_invalid_options_even_without_candidates(self):
        self.candidates = set()
        for options in ({'weights': (0, 0, 0)}, {'top_k': 0}, {'top_k': True}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                self.rank(**options)

    def test_no_mutation(self):
        papers = self.papers.copy(deep=True)
        graph, embeddings, candidates = deepcopy((self.graph, self.embeddings, self.candidates))
        result = self.rank()
        result['ranking'][0]['weighted_similarity'] = 99.
        result['eligible_ids'].clear()
        result['excluded'].clear()
        assert_frame_equal(self.papers, papers)
        self.assertTrue(nx.utils.graphs_equal(self.graph, graph))
        self.assertEqual(self.embeddings, embeddings)
        self.assertEqual(self.candidates, candidates)

    def test_no_provider_call(self):
        with patch('literature_recommender.embeddings.embed_texts', side_effect=AssertionError('API called')), \
             patch('openai.resources.embeddings.Embeddings.create', side_effect=AssertionError('API called')):
            self.rank()


if __name__ == '__main__':
    unittest.main()
