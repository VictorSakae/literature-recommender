"""Small deterministic held-out fixtures; no real dataset or external service."""

from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from literature_recommender.evaluation import evaluate_ranking
from literature_recommender.final_evaluation import (
    CONTROLLED, EXPECTED_OPTIMIZED, METRICS, REPRESENTATIONS, WEIGHT_COLUMNS,
    load_best_weights, prepare_evaluation_tables, read_evaluation_scores,
    run_final_evaluation,
)
from literature_recommender.materialization import COMMON, TEXTUAL, file_hash


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        common, textual = [], []
        for q in ('q1', 'q2'):
            for i in range(25):
                p = f'c{i:02}'
                relevant = i in (0, 2, 5, 20) if q == 'q1' else i == 24
                score = (25 - i) / 25
                common.append(('evaluation', q, p, relevant, score, score, i != 12))
                # Center value equals the mean of all other abstract scores.
                textual.append(('evaluation', q, p, score, i == 12))
        self.common = pd.DataFrame(common, columns=COMMON)
        self.representations = {n: pd.DataFrame(textual, columns=TEXTUAL) for n in REPRESENTATIONS}
        self.queries = pd.DataFrame(dict(query_id=['q1', 'q2'], n_eligible_candidates=[25, 25],
            n_ground_truth=[4, 1], missing_abstract_fraction=[.04, .04]))
        self.best = {n: dict(canonical_best_weights=list(weights),
                             best_mean_recall_at_20=mean)
                     for (n, weights), mean in zip(EXPECTED_OPTIMIZED.items(), [.8, .7, .6])}

    def run_fixture(self):
        return run_final_evaluation(self.common, self.representations, self.queries, self.best)

    def test_integrated_manual_metrics_zero_preservation_and_top20(self):
        query, top20, summary, results, checked = self.run_fixture()
        self.assertEqual(len(query), 12)
        self.assertEqual(len(top20), 240)
        self.assertEqual(len(summary), 6)
        self.assertEqual(checked, ['q1', 'q2'])
        self.assertEqual(set(query.query_id), {'q1', 'q2'})
        self.assertTrue(query.n_candidates.eq(25).all())
        self.assertTrue(query.missing_abstract_fraction.eq(.04).all())
        for (condition, name), group in query.groupby(['condition', 'representation']):
            self.assertEqual(set(group.query_id), {'q1', 'q2'})
            q1, q2 = group.set_index('query_id').loc[['q1', 'q2']].to_dict('records')
            self.assertEqual(q1['n_ground_truth'], 4)
            self.assertEqual(q2['n_ground_truth'], 1)
            for metric, expected in [('hit_at_1', 1.), ('hit_at_5', .4),
                                     ('hit_at_10', .3), ('recall_at_20', .75)]:
                self.assertEqual(q1[metric], expected)
                self.assertEqual(q2[metric], 0.)
            weights = CONTROLLED if condition == 'controlled' else EXPECTED_OPTIMIZED[name]
            self.assertEqual(tuple(q1[c] for c in WEIGHT_COLUMNS), weights)
        for _, group in top20.groupby(['condition', 'representation', 'query_id']):
            self.assertEqual(group['rank'].tolist(), list(range(1, 21)))
            self.assertEqual(group.candidate_id.tolist(), [f'c{i:02}' for i in range(20)])
            self.assertTrue(group.loc[group.candidate_id.eq('c12'), 'abstract_imputed'].all())
        self.assertEqual(results['optimized_vs_controlled_deltas']['embedding']['delta_mean_recall_at_20'], 0.)
        gap = results['tuning_to_evaluation_comparison']['embedding']
        self.assertEqual(gap['tuning_to_evaluation_difference'], .375 - .8)

    def test_aggregation_mean_population_std_median_min_max(self):
        _, _, summary, _, _ = self.run_fixture()
        self.assertTrue(summary.n_queries.eq(2).all())
        for metric, high in [('hit_at_1', 1.), ('hit_at_5', .4),
                             ('hit_at_10', .3), ('recall_at_20', .75)]:
            for stat in ('mean', 'std', 'median'):
                self.assertTrue(summary[f'{stat}_{metric}'].eq(high / 2).all())
            self.assertTrue(summary[f'min_{metric}'].eq(0).all())
            self.assertTrue(summary[f'max_{metric}'].eq(high).all())

    def test_weights_read_from_artifact_and_validated(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'best.json'
            path.write_text(json.dumps(self.best))
            self.assertEqual(load_best_weights(path), self.best)
            changed = deepcopy(self.best)
            changed['embedding']['canonical_best_weights'] = [.6, .2, .2]
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Frozen optimized weights differ'):
                load_best_weights(path)
            with self.assertRaisesRegex(ValueError, 'Frozen optimized weights differ'):
                run_final_evaluation(self.common, self.representations, self.queries, changed)

    def test_determinism_input_order_independence_and_nonmutation(self):
        before = deepcopy((self.common, self.representations, self.queries, self.best))
        first = self.run_fixture()
        second = run_final_evaluation(self.common.iloc[::-1],
            {n: f.iloc[::-1] for n, f in self.representations.items()}, self.queries.iloc[::-1], self.best)
        for i in range(3):
            pd.testing.assert_frame_equal(first[i], second[i], check_exact=True)
        self.assertEqual(first[3:], second[3:])
        pd.testing.assert_frame_equal(self.common, before[0])
        for name in REPRESENTATIONS:
            pd.testing.assert_frame_equal(self.representations[name], before[1][name])
        pd.testing.assert_frame_equal(self.queries, before[2])
        self.assertEqual(self.best, before[3])

    def test_poisoned_tuning_is_discarded_before_conversion_and_metrics(self):
        with TemporaryDirectory() as directory:
            frames = {}
            for name, frame in {'common': self.common, **self.representations}.items():
                path = Path(directory) / f'{name}.csv'
                frame.to_csv(path, index=False)
                with path.open('a') as source:
                    source.write('tuning,never_evaluate,evil,' + ','.join(
                        ['poison'] * (len(frame.columns) - 3)) + '\n')
                digest = file_hash(path)
                frames[name] = read_evaluation_scores(path, list(frame.columns))
                self.assertEqual(file_hash(path), digest)
            with patch('literature_recommender.final_evaluation.evaluate_ranking',
                       wraps=evaluate_ranking) as metric:
                query, _, _, _, _ = run_final_evaluation(frames.pop('common'), frames, self.queries, self.best)
            self.assertEqual(metric.call_count, 12)
            self.assertEqual(set(query.query_id), {'q1', 'q2'})

    def test_tuning_rows_rejected_at_evaluation_boundary(self):
        common = self.common.copy()
        common.loc[0, 'split'] = 'tuning'
        with patch('literature_recommender.final_evaluation.hybrid_scores') as scorer:
            with self.assertRaisesRegex(ValueError, 'Only evaluation'):
                run_final_evaluation(common, self.representations, self.queries, self.best)
            scorer.assert_not_called()

    def test_strict_pair_matching_fails_for_missing_extra_duplicate_changed(self):
        for kind in ('missing', 'extra', 'duplicate', 'changed'):
            reps = deepcopy(self.representations)
            frame = reps['embedding']
            if kind == 'missing':
                frame = frame.iloc[1:]
            elif kind == 'extra':
                frame = pd.concat([frame, frame.iloc[:1].assign(candidate_id='extra')])
            elif kind == 'duplicate':
                frame = pd.concat([frame, frame.iloc[:1]])
            else:
                frame.loc[0, 'candidate_id'] = 'changed'
            reps['embedding'] = frame
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                prepare_evaluation_tables(self.common, reps, self.queries)

    def test_protocol_ids_counts_truth_missingness_nonfinite(self):
        for kind in ('ids', 'count', 'truth', 'missingness', 'empty_truth', 'nonfinite', 'small_pool'):
            common, reps, queries = deepcopy((self.common, self.representations, self.queries))
            if kind == 'ids':
                queries.loc[0, 'query_id'] = 'wrong'
            elif kind == 'count':
                queries.loc[0, 'n_eligible_candidates'] = 26
            elif kind == 'truth':
                queries.loc[0, 'n_ground_truth'] = 5
            elif kind == 'missingness':
                queries.loc[0, 'missing_abstract_fraction'] = .5
            elif kind == 'empty_truth':
                common.relevant = False
            elif kind == 'nonfinite':
                reps['embedding'].loc[0, 'abstract_similarity'] = np.inf
            else:
                common = common[common.candidate_id.lt('c19')]
                reps = {n: f[f.candidate_id.lt('c19')] for n, f in reps.items()}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                run_final_evaluation(common, reps, queries, self.best)

    def test_csv_persistence_round_trip_schema(self):
        query, top, summary, results, _ = self.run_fixture()
        with TemporaryDirectory() as directory:
            for name, frame in [('query', query), ('top', top), ('summary', summary)]:
                path = Path(directory) / f'{name}.csv'
                frame.to_csv(path, index=False)
                restored = pd.read_csv(path, float_precision='round_trip')
                pd.testing.assert_frame_equal(frame, restored, check_exact=True)
            path = Path(directory) / 'results.json'
            path.write_text(json.dumps(results, allow_nan=False))
            self.assertEqual(json.loads(path.read_text()), results)

    def test_command_hash_mismatch_and_no_overwrite(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_frozen_experiment.py'
        spec = importlib.util.spec_from_file_location('evaluate_command', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as directory, ExitStack() as stack:
            previous = Path.cwd()
            os.chdir(directory)
            stack.callback(os.chdir, previous)
            output = Path('data/processed')
            output.mkdir(parents=True)
            frozen = output / 'input.csv'
            frozen.write_text('intact')
            digest = file_hash(frozen)
            stack.enter_context(patch.object(module, 'FROZEN_HASHES', {'input.csv': 'wrong'}))
            runner = stack.enter_context(patch.object(module, 'run_final_evaluation'))
            with self.assertRaisesRegex(ValueError, 'Frozen source/artifact changed'):
                module.main()
            runner.assert_not_called()
            self.assertEqual(file_hash(frozen), digest)
            existing = output / 'evaluation_summary.csv'
            existing.write_text('preserve')
            with self.assertRaises(FileExistsError):
                module.main()
            self.assertEqual(existing.read_text(), 'preserve')

    def test_loader_rejects_unknown_split_or_schema(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'scores.csv'
            self.common.assign(split='typo').to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Unexpected split'):
                read_evaluation_scores(path, COMMON)
            self.common.drop(columns='relevant').to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Unexpected score schema'):
                read_evaluation_scores(path, COMMON)


if __name__ == '__main__':
    unittest.main()
