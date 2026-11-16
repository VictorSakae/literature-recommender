"""Manually checkable metrics, without datasets or recommendation machinery."""

import unittest

from literature_recommender.evaluation import evaluate_ranking, hit_at_k, recall_at_k


class EvaluationTests(unittest.TestCase):
    def test_five_hits(self):
        self.assertEqual(hit_at_k(list('ABCDE'), set('ABCDE'), 5), 1.0)

    def test_three_hits(self):
        self.assertEqual(hit_at_k(list('ABCDE'), set('ACE'), 5), 0.6)

    def test_zero_hits(self):
        self.assertEqual(hit_at_k(list('ABCDE'), {'X'}, 5), 0.0)

    def test_paper_two_references_example(self):
        self.assertEqual(hit_at_k(list('AXBYZ'), {'A', 'B'}, 5), 0.4)

    def test_recall_three_of_four(self):
        ranking = list('ABC') + [f'X{i}' for i in range(17)]
        self.assertEqual(recall_at_k(ranking, set('ABCD'), 20), 0.75)

    def test_recall_all(self):
        self.assertEqual(recall_at_k(list('AXBYZ'), {'A', 'B'}, 5), 1.0)

    def test_recall_none(self):
        self.assertEqual(recall_at_k(list('XYZ'), {'A', 'B'}, 3), 0.0)

    def test_many_irrelevant_and_boundary(self):
        ranking = [f'X{i}' for i in range(19)] + ['A', 'B']
        self.assertEqual(recall_at_k(ranking, {'A', 'B'}, 20), 0.5)
        self.assertEqual(hit_at_k(ranking, {'A', 'B'}, 20), 0.05)

    def test_empty_ground_truth_is_ineligible_for_all_metrics(self):
        ranking = list(map(str, range(20)))
        for metric in (hit_at_k, recall_at_k):
            with self.subTest(metric=metric), self.assertRaisesRegex(ValueError, 'ground truth'):
                metric(ranking, set(), 5)
        with self.assertRaisesRegex(ValueError, 'ground truth'):
            evaluate_ranking(ranking, {})

    def test_nonpositive_k(self):
        for metric in (hit_at_k, recall_at_k):
            for k in (0, -1):
                with self.subTest(metric=metric, k=k), self.assertRaises(ValueError):
                    metric(['A'], {'A'}, k)

    def test_noninteger_k(self):
        for metric in (hit_at_k, recall_at_k):
            for k in (1.0, '1', None, True, False):
                with self.subTest(metric=metric, k=k), self.assertRaises(TypeError):
                    metric(['A'], {'A'}, k)

    def test_short_ranking(self):
        for metric in (hit_at_k, recall_at_k):
            for ranking in ([], list('ABCD')):
                with self.subTest(metric=metric, ranking=ranking), self.assertRaisesRegex(ValueError, 'at least k'):
                    metric(ranking, {'A'}, 5)
        with self.assertRaisesRegex(ValueError, 'at least k'):
            evaluate_ranking(list(map(str, range(19))), {'0'})

    def test_duplicates_including_beyond_cutoff(self):
        for metric in (hit_at_k, recall_at_k):
            for ranking in (list('AAB'), list('ABCA')):
                with self.subTest(metric=metric, ranking=ranking), self.assertRaisesRegex(ValueError, 'duplicate'):
                    metric(ranking, {'A'}, 2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluate_ranking(list(map(str, range(20))) + ['0'], {'0'})

    def test_relevant_duplicates_have_set_semantics(self):
        self.assertEqual(hit_at_k(list('AXBYZ'), list('AABB'), 5), 0.4)
        self.assertEqual(recall_at_k(list('AXBYZ'), list('AABB'), 5), 1.0)

    def test_combined_manually_calculated(self):
        # Hits at positions 1, 3, 5, 10, 20; F is absent; G occurs after top-20.
        ranking = [f'X{i}' for i in range(21)]
        for position, paper_id in zip((0, 2, 4, 9, 19, 20), 'ABCDEG'):
            ranking[position] = paper_id
        self.assertEqual(evaluate_ranking(ranking, set('ABCDEFG')), {
            'hit@1': 1.0, 'hit@5': 0.6, 'hit@10': 0.4, 'recall@20': 5 / 7,
        })

    def test_iterators_consumed_once(self):
        ranking = list(map(str, range(20)))
        expected = {'hit@1': 1.0, 'hit@5': 0.2, 'hit@10': 0.1, 'recall@20': 1.0}
        self.assertEqual(evaluate_ranking(iter(ranking), iter(['0', '0'])), expected)
        self.assertEqual(hit_at_k(iter(ranking), iter(['0']), 5), 0.2)
        self.assertEqual(recall_at_k(iter(ranking), iter(['0']), 20), 1.0)

    def test_inputs_not_modified(self):
        ranking, relevant = list(map(str, range(20))), ['0', '1', '1']
        before = ranking.copy(), relevant.copy()
        hit_at_k(ranking, relevant, 5)
        recall_at_k(ranking, relevant, 20)
        evaluate_ranking(ranking, relevant).clear()
        self.assertEqual((ranking, relevant), before)


if __name__ == '__main__':
    unittest.main()
