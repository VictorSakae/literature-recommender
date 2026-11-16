"""Small synthetic checks of ground truth, eligibility and readiness preparation."""

from copy import deepcopy
from random import Random
import unittest
from unittest.mock import patch

import networkx as nx
import pandas as pd
from pandas.testing import assert_frame_equal

from literature_recommender.experiment import (
    available_references, prepare_reference_query, reference_query_population,
    select_query_sample,
)
from literature_recommender.evaluation import evaluate_ranking
from literature_recommender.recommender import rank_candidate_list


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        ids = ['Q', 'incoming', 'isolated', 'noabstract', 'self', 'externalonly']
        ids += [f'C{i:02}' for i in range(21)]
        self.papers = pd.DataFrame({
            'paper_id': ids, 'title': ['valid title'] * len(ids),
            'abstract': ['same abstract'] * len(ids),
        }, dtype='string')
        self.papers.loc[self.papers.paper_id.isin(['C00', 'noabstract']), 'abstract'] = pd.NA
        self.graph = nx.DiGraph([
            ('Q', 'Q'), ('Q', 'C00'), ('Q', 'C01'), ('Q', 'external'),
            ('incoming', 'Q'), ('noabstract', 'Q'), ('self', 'self'),
            ('externalonly', 'external'),
        ])
        self.graph.add_edges_from(('C01', f'C{i:02}') for i in range(2, 21))

    def test_truth_only_direct_available_references(self):
        metadata = set(self.papers.paper_id)
        self.assertEqual(available_references(self.graph, 'Q', metadata), {'C00', 'C01'})
        # C00 is relevant despite its missing abstract; depth 2 and incoming are not.
        self.assertEqual(available_references(self.graph, 'self', metadata), set())
        self.assertEqual(available_references(self.graph, 'externalonly', metadata), set())
        with self.assertRaises(KeyError):
            available_references(self.graph, 'absent', metadata)

    def test_complete_pool_and_diagnostic(self):
        plan = prepare_reference_query(self.papers, self.graph, 'Q')
        self.assertEqual(plan['relevant_ids'], {'C00', 'C01'})
        self.assertEqual(len(plan['candidate_ids']), 22)
        self.assertEqual(len(plan['eligible_ids']), 21)
        self.assertEqual(plan['excluded'], {'external': 'missing_metadata'})
        self.assertIsNone(plan['abstracts']['C00'])
        self.assertEqual(plan['diagnostic'], {
            'query_id': 'Q', 'n_ground_truth': 2, 'n_structural_candidates': 22,
            'n_eligible_candidates': 21, 'n_candidates_without_abstract': 1,
            'n_candidates_without_metadata': 1, 'n_unique_abstracts_needed': 1,
        })

    def test_population_criteria(self):
        population = reference_query_population(self.papers, self.graph)
        self.assertEqual(set(population.query_id), {'Q', 'incoming'})
        self.assertEqual(population.set_index('query_id').loc['Q', 'n_eligible_candidates'], 21)
        self.graph.remove_edge('C01', 'C20')
        self.assertIn('Q', set(reference_query_population(self.papers, self.graph).query_id))
        self.graph.remove_edge('C01', 'C19')
        self.assertNotIn('Q', set(reference_query_population(self.papers, self.graph).query_id))

    def test_synthetic_reconstruction_through_existing_ranker(self):
        plan = prepare_reference_query(self.papers, self.graph, 'Q')
        truth_before = plan['relevant_ids'].copy()
        # Synthetic unit fixture only; never used for the real-data diagnostic.
        vectors = {paper_id: None if text is None else [1., 0.]
                   for paper_id, text in plan['abstracts'].items()}
        result = rank_candidate_list(self.papers, self.graph, 'Q',
                                     plan['candidate_ids'], vectors)
        self.assertEqual(len(result['ranking']), 21)
        self.assertEqual(result['excluded'], plan['excluded'])
        metrics = evaluate_ranking([row['paper_id'] for row in result['ranking']],
                                   plan['relevant_ids'])
        self.assertEqual(metrics, {'hit@1': 1., 'hit@5': .4,
                                   'hit@10': .2, 'recall@20': 1.})
        self.assertEqual(plan['relevant_ids'], truth_before)

    def test_ineligible_queries_fail_preparation(self):
        for query, error in [('isolated', KeyError), ('absent', KeyError),
                             ('noabstract', ValueError), ('self', ValueError),
                             ('externalonly', ValueError), ('C01', ValueError)]:
            with self.subTest(query=query), self.assertRaises(error):
                prepare_reference_query(self.papers, self.graph, query)

    def test_no_available_candidate_abstracts(self):
        self.papers.loc[self.papers.paper_id.str.startswith('C'), 'abstract'] = pd.NA
        self.assertNotIn('Q', set(reference_query_population(self.papers, self.graph).query_id))
        with self.assertRaisesRegex(ValueError, 'imputation'):
            prepare_reference_query(self.papers, self.graph, 'Q')

    def test_duplicate_metadata_fails(self):
        papers = pd.concat([self.papers, self.papers.iloc[:1]])
        for operation in (lambda: reference_query_population(papers, self.graph),
                          lambda: prepare_reference_query(papers, self.graph, 'Q')):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                operation()

    def test_seed_order_and_cost_cap_before_sampling(self):
        population = pd.DataFrame({'query_id': list('HGFEDCBA'),
                                   'n_eligible_candidates': [101, 19, 100, 90, 80, 70, 60, 20]})
        expected = Random(42).sample(list('ABCDEF'), 5)
        self.assertEqual(select_query_sample(population, seed=42, max_candidates=100), expected)
        self.assertEqual(select_query_sample(population.iloc[::-1], seed=42, max_candidates=100), expected)
        self.assertEqual(select_query_sample(population, seed=42, max_candidates=60), ['A', 'B'])
        self.assertEqual(select_query_sample(population.iloc[:0], seed=42), [])

    def test_invalid_sample_options(self):
        population = reference_query_population(self.papers, self.graph)
        for options in ({'max_queries': 0}, {'max_queries': 6}, {'max_queries': True},
                        {'max_candidates': 19}, {'max_candidates': 20.0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                select_query_sample(population, seed=42, **options)
        with self.assertRaises(ValueError):
            select_query_sample(pd.concat([population, population]), seed=42)

    def test_no_mutation_or_provider_calls(self):
        papers, graph = self.papers.copy(deep=True), deepcopy(self.graph)
        with patch('openai.resources.embeddings.Embeddings.create', side_effect=AssertionError('API called')):
            population = reference_query_population(self.papers, self.graph)
            before = population.copy(deep=True)
            select_query_sample(population, seed=42)
            plan = prepare_reference_query(self.papers, self.graph, 'Q')
            plan['relevant_ids'].clear()
            plan['candidate_ids'].clear()
            plan['abstracts'].clear()
        assert_frame_equal(self.papers, papers)
        assert_frame_equal(population, before)
        self.assertTrue(nx.utils.graphs_equal(self.graph, graph))


if __name__ == '__main__':
    unittest.main()
