"""Synthetic tuning-only searches: 2 queries, 21 candidates each, no providers."""

from copy import deepcopy
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from literature_recommender.grid_search import (
    REPRESENTATIONS, WEIGHT_COLUMNS, prepare_tuning_tables, read_tuning_scores,
    regression_checks, run_grid_search, select_best, weight_grid,
)
from literature_recommender.materialization import COMMON, TEXTUAL, file_hash
from literature_recommender.recommender import hybrid_scores, rank_scores


class GridSearchTests(unittest.TestCase):
    def setUp(self):
        common, abstracts = [], {name: [] for name in REPRESENTATIONS}
        for q in ('q1', 'q2'):
            for p in [f'a{i:02}' for i in range(20)] + ['z_relevant']:
                relevant = p == 'z_relevant'
                title = float(q == 'q1') if relevant else .5
                citation = 0. if relevant else .5
                common.append(('tuning', q, p, relevant, title, citation, True))
                for name, q1 in [('embedding', 1.), ('tfidf', 0.), ('word2vec', .5)]:
                    score = (q1 if q == 'q1' else 1.) if relevant else .5
                    abstracts[name].append(('tuning', q, p, score, False))
        self.common = pd.DataFrame(common, columns=COMMON)
        self.representations = {n: pd.DataFrame(rows, columns=TEXTUAL)
                                for n, rows in abstracts.items()}
        self.queries = pd.DataFrame(dict(query_id=['q1', 'q2'],
                                        n_eligible_candidates=[21, 21], n_ground_truth=[1, 1]))

    def test_grid_simplex_and_inclusions(self):
        grid = weight_grid()
        self.assertEqual(len(grid), 66)
        self.assertEqual(len(set(grid)), 66)
        for weights in grid:
            self.assertTrue(all(w >= 0 for w in weights))
            self.assertAlmostEqual(sum(weights), 1., places=15)
        for weights in [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (.6, .2, .2)]:
            self.assertIn(weights, grid)

    def test_manual_weighted_score_extremes_and_tie_break(self):
        scores = hybrid_scores({'z': .8, 'a': .8}, {'z': .5, 'a': .5},
                               {'z': .3, 'a': .3}, weights=(.6, .2, .2))
        self.assertEqual(scores['z'], .6 * .8 + .2 * .5 + .2 * .3)
        self.assertEqual([p for p, _ in rank_scores(scores)], ['a', 'z'])
        tables = prepare_tuning_tables(self.common, self.representations, self.queries)
        self.assertEqual(regression_checks(tables)['embedding'], ['q1', 'q2'])

    def test_integrated_full_grid_known_optima_independent_representations(self):
        results, individual, best, _ = run_grid_search(self.common, self.representations, self.queries)
        self.assertEqual(len(results), 198)
        self.assertEqual(len(individual), 396)
        self.assertTrue(results.n_queries.eq(2).all())
        expected = dict(embedding=([.6, 0., .4], 1.),
                        tfidf=([0., .6, .4], .5), word2vec=([.6, .3, .1], 1.))
        for name, (weights, mean) in expected.items():
            self.assertEqual(len(results[results.representation.eq(name)]), 66)
            self.assertEqual(best[name]['canonical_best_weights'], weights)
            self.assertEqual(best[name]['best_mean_recall_at_20'], mean)
            configs = best[name]['all_best_configs']
            self.assertGreater(len(configs), 1)
            self.assertEqual(best[name]['n_best_tied_configs'], len(configs))
            self.assertEqual(configs, sorted(configs))
        # Equal query weighting, independently from the persisted individual values.
        for row in results.itertuples(index=False):
            selected = individual.loc[individual.representation.eq(row.representation)]
            for c in WEIGHT_COLUMNS:
                selected = selected.loc[selected[c].eq(getattr(row, c))]
            self.assertEqual(row.mean_recall_at_20, selected.recall_at_20.mean())

    def test_tolerance_retains_all_near_maxima_without_secondary_metrics(self):
        rows = []
        for name in REPRESENTATIONS:
            for weights, mean, diagnostic in [((0., 0., 1.), .5 - 5e-13, 999),
                                               ((0., .1, .9), .5, 0),
                                               ((0., .2, .8), .5 - 2e-12, 1000)]:
                rows.append((name, *weights, mean, diagnostic))
        results = pd.DataFrame(rows, columns=['representation', *WEIGHT_COLUMNS,
                                             'mean_recall_at_20', 'std_recall_at_20'])
        for info in select_best(results).values():
            self.assertEqual(info['n_best_tied_configs'], 2)
            self.assertEqual(info['all_best_configs'], [[0., 0., 1.], [0., .1, .9]])
            self.assertEqual(info['canonical_best_weights'], [0., 0., 1.])
            self.assertEqual(info['best_mean_recall_at_20'], .5)

    def test_determinism_order_independence_and_nonmutation(self):
        before = deepcopy((self.common, self.representations, self.queries))
        first = run_grid_search(self.common, self.representations, self.queries)
        second = run_grid_search(self.common.iloc[::-1],
                                 {n: f.iloc[::-1] for n, f in self.representations.items()},
                                 self.queries.iloc[::-1])
        for i in (0, 1):
            pd.testing.assert_frame_equal(first[i], second[i], check_exact=True)
        self.assertEqual(first[2:], second[2:])
        pd.testing.assert_frame_equal(self.common, before[0])
        for n in REPRESENTATIONS:
            pd.testing.assert_frame_equal(self.representations[n], before[1][n])
        pd.testing.assert_frame_equal(self.queries, before[2])

    def test_streaming_loader_ignores_poisoned_evaluation_and_preserves_files(self):
        with TemporaryDirectory() as directory:
            common_path = Path(directory) / 'common.csv'
            self.common.to_csv(common_path, index=False)
            with common_path.open('a') as f:
                f.write('evaluation,never_rank_this,evil,poison,poison,poison,poison\n')
            digest = file_hash(common_path)
            common = read_tuning_scores(common_path, COMMON)
            representations = {}
            for name, frame in self.representations.items():
                path = Path(directory) / f'{name}.csv'
                frame.to_csv(path, index=False)
                with path.open('a') as f:
                    f.write('evaluation,never_rank_this,evil,poison,poison\n')
                representations[name] = read_tuning_scores(path, TEXTUAL)
            with patch('literature_recommender.grid_search.recall_at_k',
                       wraps=__import__('literature_recommender.evaluation',
                                        fromlist=['recall_at_k']).recall_at_k) as metric:
                result = run_grid_search(common, representations, self.queries)
            self.assertEqual(metric.call_count, 396)
            self.assertEqual(set(result[1].query_id), {'q1', 'q2'})
            self.assertEqual(file_hash(common_path), digest)

    def test_evaluation_rows_rejected_by_search_boundary(self):
        changed = self.common.copy()
        changed.loc[0, 'split'] = 'evaluation'
        with patch('literature_recommender.grid_search.hybrid_scores') as scorer:
            with self.assertRaisesRegex(ValueError, 'Only tuning'):
                run_grid_search(changed, self.representations, self.queries)
            scorer.assert_not_called()

    def test_missing_extra_duplicate_and_changed_candidate_fail(self):
        for kind in ('missing', 'extra', 'duplicate', 'changed'):
            reps = deepcopy(self.representations)
            frame = reps['embedding']
            if kind == 'missing':
                frame = frame.iloc[1:]
            elif kind == 'extra':
                extra = frame.iloc[:1].assign(candidate_id='extra')
                frame = pd.concat([frame, extra])
            elif kind == 'duplicate':
                frame = pd.concat([frame, frame.iloc[:1]])
            else:
                frame.loc[0, 'candidate_id'] = 'changed'
            reps['embedding'] = frame
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                prepare_tuning_tables(self.common, reps, self.queries)

    def test_invalid_protocol_ids_counts_truth_and_nonfinite_fail(self):
        for kind in ('ids', 'counts', 'truth', 'small_pool', 'empty_truth', 'nonfinite'):
            common, reps, queries = deepcopy((self.common, self.representations, self.queries))
            if kind == 'ids':
                queries.loc[0, 'query_id'] = 'wrong'
            elif kind == 'counts':
                queries.loc[0, 'n_eligible_candidates'] = 22
            elif kind == 'truth':
                queries.loc[0, 'n_ground_truth'] = 2
            elif kind == 'small_pool':
                common = common[common.candidate_id.ne('a00') & common.candidate_id.ne('a01')]
                reps = {n: f[f.candidate_id.ne('a00') & f.candidate_id.ne('a01')]
                        for n, f in reps.items()}
            elif kind == 'empty_truth':
                common.relevant = False
            else:
                reps['embedding'].loc[0, 'abstract_similarity'] = np.inf
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                run_grid_search(common, reps, queries)

    def test_command_stops_on_hash_mismatch_before_search(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/grid_search_weights.py'
        spec = importlib.util.spec_from_file_location('grid_command', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as directory, ExitStack() as stack:
            previous = Path.cwd()
            os.chdir(directory)
            stack.callback(os.chdir, previous)
            output = Path('data/processed')
            output.mkdir(parents=True)
            frozen = output / 'frozen.csv'
            frozen.write_text('unchanged')
            digest = file_hash(frozen)
            stack.enter_context(patch.object(module, 'FROZEN_HASHES', {'frozen.csv': 'wrong'}))
            search = stack.enter_context(patch.object(module, 'run_grid_search'))
            with self.assertRaisesRegex(ValueError, 'Frozen source/artifact changed'):
                module.main()
            search.assert_not_called()
            self.assertEqual(file_hash(frozen), digest)
            self.assertEqual(list(output.iterdir()), [frozen])

    def test_loader_rejects_unknown_split_and_schema(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.csv'
            self.common.assign(split='typo').to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Unexpected split'):
                read_tuning_scores(path, COMMON)
            self.common.drop(columns='relevant').to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Unexpected score schema'):
                read_tuning_scores(path, COMMON)


if __name__ == '__main__':
    unittest.main()
