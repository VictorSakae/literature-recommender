"""Stage 7A: manual synthetic graph; no similarity, ranking or metric calls."""
from copy import deepcopy
from random import Random
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from gensim.models import KeyedVectors

from literature_recommender.experiment import (
    build_experiment_population, split_experiment_queries,
    experiment_query_diagnostics, validate_fixed_textual_support,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        candidates = [f'C{i:02}' for i in range(21)]
        ids = ['Q', 'R', 'S', 'outside', 'missing', 'empty', 'few', 'invalid'] + candidates
        self.papers = pd.DataFrame(dict(paper_id=ids, title=['valid title'] * len(ids),
                                        abstract=['valid abstract'] * len(ids)), dtype='string')
        self.papers.loc[self.papers.paper_id.isin(['C00', 'missing']), 'abstract'] = pd.NA
        self.graph = nx.DiGraph()
        self.graph.add_edges_from((q, 'C01') for q in ['Q', 'R', 'S', 'missing', 'invalid'])
        self.graph.add_edges_from(('C01', c) for c in candidates if c != 'C01')
        self.graph.add_edges_from([('Q', 'external'), ('Q', 'Q'), ('empty', 'empty'), ('few', 'C20')])
        self.valid = set(ids) - {'C00', 'missing', 'invalid'}

    def population(self):
        return build_experiment_population(self.papers, self.graph, textual_valid_ids=self.valid)

    def test_manual_eligibility_and_reasons(self):
        rows = self.population().set_index('query_id')
        self.assertEqual(set(rows.index[rows.eligible]), {'Q', 'R', 'S', 'C01'})
        for p, reason in dict(external='NO_METADATA', outside='NOT_IN_GRAPH',
                              missing='NO_QUERY_ABSTRACT', empty='EMPTY_GROUND_TRUTH',
                              few='TOO_FEW_ELIGIBLE_CANDIDATES',
                              invalid='NO_VALID_TEXTUAL_COMPARISON').items():
            self.assertEqual(rows.loc[p, 'exclusion_reason'], reason)
        q = rows.loc['Q']
        self.assertEqual(q.n_ground_truth, 1)  # external/self/depth2 are not truth
        self.assertEqual(q.n_structural_candidates, 22)
        self.assertEqual(q.n_eligible_candidates, 21)  # not truncated
        self.assertEqual(q.n_candidates_with_abstract, 20)
        self.assertEqual(q.n_candidates_without_abstract, 1)
        self.assertAlmostEqual(q.missing_abstract_fraction, 1 / 21)

    def test_all_missing_candidates(self):
        self.papers.loc[self.papers.paper_id.str.startswith('C'), 'abstract'] = pd.NA
        self.assertEqual(self.population().set_index('query_id').loc['Q', 'exclusion_reason'],
                         'NO_VALID_TEXTUAL_COMPARISON')

    def test_unrepresentable_available_candidates_are_not_imputed(self):
        self.valid -= {'C20'}
        self.assertEqual(self.population().set_index('query_id').loc['Q', 'exclusion_reason'],
                         'INVALID_CANDIDATE_REPRESENTATION')

    def test_high_missingness_remains_eligible(self):
        self.papers.loc[self.papers.paper_id.str.startswith('C') &
                        self.papers.paper_id.ne('C01'), 'abstract'] = pd.NA
        q = self.population().set_index('query_id').loc['Q']
        self.assertTrue(q.eligible)
        self.assertEqual(q.n_candidates_with_abstract, 1)
        self.assertAlmostEqual(q.missing_abstract_fraction, 20 / 21)

    def test_no_representable_candidate_even_with_available_abstracts(self):
        self.valid -= {f'C{i:02}' for i in range(21)}
        q = self.population().set_index('query_id').loc['Q']
        self.assertEqual(q.exclusion_reason, 'NO_VALID_TEXTUAL_COMPARISON')

    def test_determinism_order_sizes_disjoint_and_no_mutation(self):
        pop = self.population()
        before = pop.copy(deep=True)
        first = split_experiment_queries(pop, n_tuning=2, n_evaluation=2)
        self.assertEqual(first, split_experiment_queries(pop.iloc[::-1], n_tuning=2, n_evaluation=2))
        rng = Random(42)
        tuning = rng.sample(sorted(pop.loc[pop.eligible].query_id), 2)
        remaining = sorted(set(pop.loc[pop.eligible].query_id) - set(tuning))
        self.assertEqual(first['tuning_query_ids'], tuning)
        self.assertEqual(first['evaluation_query_ids'], rng.sample(remaining, 2))
        self.assertEqual(len(first['evaluation_query_ids']), 2)
        self.assertFalse(set(tuning) & set(first['evaluation_query_ids']))
        other = split_experiment_queries(pop, n_tuning=2, n_evaluation=2, seed=19)
        self.assertNotEqual(first['tuning_query_ids'], other['tuning_query_ids'])
        assert_frame_equal(before, pop)

    def test_errors_and_bounds(self):
        pop = self.population()
        for options in [dict(n_tuning=3, n_evaluation=2), dict(n_tuning=0, n_evaluation=1),
                        dict(n_tuning=True, n_evaluation=1),
                        dict(n_tuning=1, n_evaluation=1, candidate_max=19)]:
            with self.assertRaises(ValueError):
                split_experiment_queries(pop, **options)
        with self.assertRaises(ValueError):
            split_experiment_queries(pd.concat([pop, pop]), n_tuning=1, n_evaluation=1)
        with self.assertRaises(ValueError):
            build_experiment_population(pd.concat([self.papers, self.papers]), self.graph,
                                        textual_valid_ids=self.valid)
        with self.assertRaises(ValueError):
            build_experiment_population(self.papers, self.graph, textual_valid_ids=self.valid, candidate_min=19)

    def test_cost_and_shared_abstracts(self):
        d = experiment_query_diagnostics(self.population(), self.papers, self.graph, ['Q', 'R'])
        self.assertEqual(d['sum_eligible_candidates'], 42)
        self.assertEqual(d['n_candidates_with_abstract'], 40)
        self.assertEqual(d['n_candidates_without_abstract'], 2)
        self.assertEqual(d['n_unique_abstracts_needed'], 1)  # same text, distinct papers
        self.assertEqual(d['n_unique_papers_with_abstract'], 22)
        self.assertEqual(d['ground_truth']['min'], 1)

    def test_inputs_unchanged_and_no_api(self):
        before, graph = self.papers.copy(deep=True), deepcopy(self.graph)
        with patch('openai.resources.embeddings.Embeddings.create', side_effect=AssertionError('API')):
            pop = self.population()
            split_experiment_queries(pop, n_tuning=1, n_evaluation=1)
            experiment_query_diagnostics(pop, self.papers, self.graph, ['Q'])
        assert_frame_equal(before, self.papers)
        self.assertTrue(nx.utils.graphs_equal(graph, self.graph))

    def test_fixed_representation_preflight(self):
        vectors = KeyedVectors(vector_size=2)
        vectors.add_vectors(['valid', 'abstract'], np.array([[1., 0.], [0., 1.]]))
        papers = pd.DataFrame(dict(paper_id=['ok', 'oov', 'absent'], title=['valid'] * 3,
                                   abstract=['valid abstract', 'unknown', pd.NA]), dtype='string')
        result = validate_fixed_textual_support(papers, vectors)
        self.assertEqual(result['valid_ids'], {'ok'})
        self.assertEqual(set(result['failures']), {'oov'})
        vectors.add_vector('unknown', np.zeros(2))
        self.assertIn('nonzero', validate_fixed_textual_support(papers, vectors)['failures']['oov'])
        papers.loc[0, 'title'] = '!'
        with self.assertRaisesRegex(ValueError, 'Invalid title'):
            validate_fixed_textual_support(papers, vectors)


if __name__ == '__main__':
    unittest.main()
