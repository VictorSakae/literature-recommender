"""Analytical unit tests, using small synthetic already-computed query metrics."""

from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from literature_recommender.analysis import (
    CONDITIONS, REPRESENTATIONS, METRICS, aligned_values, paired_difference,
    difference_summary, validate_results, recompute_summary, paired_comparisons,
    tuning_effect, correlation, correlations, quartile_labels, quartile_summaries,
    paired_bootstrap_ci,
)
from literature_recommender.final_evaluation import summarize_query_results


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.weights = dict(controlled={n: (.6, .2, .2) for n in REPRESENTATIONS},
            optimized=dict(embedding=(.6, .1, .3), tfidf=(.6, .1, .3), word2vec=(.3, .4, .3)))
        values = dict(controlled=dict(embedding=[.2, .4, .8], tfidf=[.3, .4, .6],
                                      word2vec=[.1, .5, .7]),
                      optimized=dict(embedding=[.3, .4, .8], tfidf=[.3, .5, .6], word2vec=[0., .5, .7]))
        rows = []
        for condition in CONDITIONS:
            for name in REPRESENTATIONS:
                for i, q in enumerate(['q1', 'q2', 'q3']):
                    a, t, c = self.weights[condition][name]
                    rows.append(dict(condition=condition, representation=name, query_id=q,
                        w_abstract=a, w_title=t, w_citation=c, n_candidates=[20, 40, 80][i],
                        n_ground_truth=[1, 2, 4][i], missing_abstract_fraction=[0., .4, .8][i],
                        **{m: values[condition][name][i] for m in METRICS}))
        self.results = pd.DataFrame(rows)

    def test_paired_alignment_and_differences(self):
        left = pd.Series([.8, .2, .4], index=['q3', 'q1', 'q2'])
        right = pd.Series([.3, .4, .6], index=['q1', 'q2', 'q3'])
        delta = paired_difference(left, right)
        self.assertEqual(delta.index.tolist(), ['q1', 'q2', 'q3'])
        np.testing.assert_allclose(delta, [-.1, 0., .2], atol=1e-16)
        for invalid in (right.iloc[:2], pd.Series([1., 2., 3.], index=['q1', 'q1', 'q2'])):
            with self.assertRaises(ValueError):
                aligned_values(left, invalid)

    def test_win_tie_loss_no_rounding_and_distribution(self):
        stats = difference_summary(pd.Series([-1., 0., 1., 1.]))
        self.assertEqual((stats['wins'], stats['ties'], stats['losses']), (2, 1, 1))
        self.assertEqual((stats['wins_pct'], stats['ties_pct'], stats['losses_pct']), (50., 25., 25.))
        self.assertEqual(stats['mean_difference'], .25)
        self.assertEqual(stats['median_difference'], .5)
        self.assertEqual(stats['mean_absolute_difference'], .75)
        self.assertEqual(stats['median_absolute_difference'], 1.)
        self.assertEqual(stats['p25'], -.25)
        self.assertEqual(stats['p75'], 1.)
        self.assertAlmostEqual(stats['std_difference'], np.sqrt(.6875))
        tiny = difference_summary(pd.Series([-1e-16, 0., 1e-16]))
        self.assertEqual((tiny['wins'], tiny['ties'], tiny['losses']), (1, 1, 1))

    def test_pairwise_comparisons_and_tuning_deltas(self):
        paired, diffs = paired_comparisons(self.results)
        self.assertEqual(len(paired), 24)
        self.assertEqual(len(diffs), 72)
        row = paired.loc[paired.condition.eq('controlled') & paired.comparison.eq('embedding - tfidf') &
                         paired.metric.eq('recall_at_20')].iloc[0]
        self.assertEqual((row.wins, row.ties, row.losses), (1, 1, 1))
        self.assertAlmostEqual(row.mean_difference, .1 / 3)
        tuning, deltas = tuning_effect(self.results)
        self.assertEqual(len(tuning), 12)
        self.assertEqual(len(deltas), 36)
        e = tuning.loc[tuning.comparison.eq('embedding') & tuning.metric.eq('recall_at_20')].iloc[0]
        w = tuning.loc[tuning.comparison.eq('word2vec') & tuning.metric.eq('recall_at_20')].iloc[0]
        self.assertEqual((e.improved, e.ties, e.worsened), (1, 2, 0))
        self.assertEqual((w.improved, w.ties, w.worsened), (0, 2, 1))
        self.assertAlmostEqual(e.mean_difference, 1 / 30)
        self.assertAlmostEqual(w.mean_difference, -1 / 30)

    def test_validation_pairing_weights_features_and_finite_values(self):
        validate_results(self.results, ['q1', 'q2', 'q3'], self.weights)
        for kind in ('missing', 'duplicate', 'wrong_id', 'weight', 'feature', 'nan', 'condition'):
            changed = self.results.copy()
            if kind == 'missing': changed = changed.iloc[1:]
            elif kind == 'duplicate': changed = pd.concat([changed, changed.iloc[:1]])
            elif kind == 'wrong_id': changed.loc[0, 'query_id'] = 'other'
            elif kind == 'weight': changed.loc[0, 'w_abstract'] = .5
            elif kind == 'feature': changed.loc[0, 'n_candidates'] = 21
            elif kind == 'nan': changed.loc[0, 'recall_at_20'] = np.nan
            else: changed.loc[0, 'condition'] = 'other'
            with self.subTest(kind=kind), self.assertRaises((ValueError, AssertionError)):
                validate_results(changed, ['q1', 'q2', 'q3'], self.weights)

    def test_independent_summary_matches_and_disagreement_stops(self):
        frozen = summarize_query_results(self.results)
        actual = recompute_summary(self.results, frozen)
        self.assertEqual(len(actual), 6)
        changed = frozen.copy()
        changed.loc[0, 'std_recall_at_20'] += .01
        with self.assertRaisesRegex(ValueError, 'aggregates differ'):
            recompute_summary(self.results, changed)

    def test_correlation_alignment_ranks_ties_and_constant(self):
        left = pd.Series([1., 2., 3.], index=['a', 'b', 'c'])
        right = pd.Series([9., 1., 4.], index=['c', 'a', 'b'])
        result = correlation(left, right)
        self.assertAlmostEqual(result['spearman'], 1.)
        self.assertGreater(result['pearson'], .98)
        tied = correlation(pd.Series([1., 1., 2.], index=['a', 'b', 'c']), left)
        self.assertAlmostEqual(tied['spearman'], np.sqrt(3) / 2)
        const = correlation(left * 0, right)
        self.assertIsNone(const['pearson'])
        self.assertEqual(const['status'], 'constant input')
        with self.assertRaises(ValueError): correlation(left, right.iloc[:2])

    def test_bootstrap_determinism_pairing_and_known_intervals(self):
        values = pd.Series([-1., 0., .2, .3, 1.], index=list('abcde'))
        a = paired_bootstrap_ci(values)
        b = paired_bootstrap_ci(values.iloc[::-1])
        self.assertEqual(a, b)
        self.assertEqual(a['n_resamples'], 10000)
        self.assertEqual(a['seed'], 42)
        self.assertLess(a['ci_low'], a['ci_high'])
        const = paired_bootstrap_ci(pd.Series([0., 0., 0.], index=list('abc')))
        self.assertEqual((const['ci_low'], const['ci_high']), (0., 0.))
        with self.assertRaises(ValueError): paired_bootstrap_ci(values, n_resamples=0)

    def test_feature_only_quartiles_keep_ties_and_shared_bounds(self):
        labels, bounds = quartile_labels(pd.Series([0., 0., 0., 0., 1., 1., 1., 1.]))
        self.assertEqual(bounds, [0., 0., .5, 1., 1.])
        self.assertEqual(labels.tolist(), [1, 1, 1, 1, 3, 3, 3, 3])
        _, diffs = paired_comparisons(self.results)
        quartiles = quartile_summaries(self.results, diffs)
        self.assertEqual(len(quartiles), 96)
        for _, group in quartiles.groupby(['feature', 'condition', 'outcome']):
            self.assertEqual(group.n_queries.sum(), 3)
        perf = quartiles.loc[quartiles.outcome_type.eq('recall_at_20')]
        for _, group in perf.groupby(['feature', 'quartile']):
            self.assertEqual(len(group[['lower', 'upper', 'n_queries']].drop_duplicates()), 1)

    def test_determinism_nonmutation_and_correlation_tables(self):
        before = self.results.copy(deep=True)
        p1, d1 = paired_comparisons(self.results)
        p2, d2 = paired_comparisons(self.results.iloc[::-1])
        pd.testing.assert_frame_equal(p1, p2)
        pd.testing.assert_frame_equal(d1, d2)
        c1 = correlations(self.results, d1)
        c2 = correlations(self.results.iloc[::-1], d2.iloc[::-1])
        self.assertEqual(len(c1), 30)
        pd.testing.assert_frame_equal(c1, c2)
        pd.testing.assert_frame_equal(self.results, before)

    def test_command_hash_failure_before_analysis(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/analyze_final_results.py'
        spec = importlib.util.spec_from_file_location('analysis_command', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as directory, ExitStack() as stack:
            previous = Path.cwd()
            os.chdir(directory)
            stack.callback(os.chdir, previous)
            output = Path('data/processed')
            output.mkdir(parents=True)
            path = output / 'frozen.csv'
            path.write_text('preserved')
            stack.enter_context(patch.object(module, 'FROZEN_HASHES', {'frozen.csv': 'wrong'}))
            runner = stack.enter_context(patch.object(module, 'paired_comparisons'))
            with self.assertRaisesRegex(ValueError, 'Frozen source/artifact changed'):
                module.main()
            runner.assert_not_called()
            self.assertEqual(path.read_text(), 'preserved')


if __name__ == '__main__':
    unittest.main()
