"""Hybrid combination and ranking with manually checkable synthetic scores."""

from copy import deepcopy
import unittest

from literature_recommender.recommender import (
    hybrid_scores,
    rank_scores,
    weighted_similarity,
)


class WeightedSimilarityTests(unittest.TestCase):
    def test_known_formula_and_defaults(self):
        result = weighted_similarity(0.8, 0.5, 0.3)
        self.assertEqual(result, 0.6 * 0.8 + 0.2 * 0.5 + 0.2 * 0.3)
        self.assertIs(type(result), float)

    def test_extreme_weights(self):
        for weights, expected in (((1, 0, 0), .8), ((0, 1, 0), .5), ((0, 0, 1), .3)):
            with self.subTest(weights=weights):
                self.assertEqual(weighted_similarity(.8, .5, .3, weights=weights), expected)

    def test_wrong_weight_count(self):
        for weights in ((), (1,), (.5, .5), (.4, .2, .2, .2)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                weighted_similarity(.8, .5, .3, weights=weights)

    def test_wrong_sum_or_negative_weight(self):
        for weights in ((.6, .2, .3), (0, 0, 0), (-.1, .5, .6)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                weighted_similarity(.8, .5, .3, weights=weights)

    def test_invalid_weight_types(self):
        for weights in (None, 1, 'abc', ('0.6', .2, .2), (True, 0, 0), (1j, 0, 0)):
            with self.subTest(weights=weights), self.assertRaises(TypeError):
                weighted_similarity(.8, .5, .3, weights=weights)

    def test_nonfinite_weights(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                weighted_similarity(.8, .5, .3, weights=(value, .2, .2))

    def test_tolerance_does_not_renormalize_weights(self):
        weights = (.6 + 1e-10, .2, .2)
        self.assertEqual(weighted_similarity(1., 0., 0., weights=weights), weights[0])
        with self.assertRaises(ValueError):
            weighted_similarity(1., 0., 0., weights=(.6 + 1e-8, .2, .2))

    def test_signed_scores_are_not_rounded_or_normalized(self):
        self.assertEqual(weighted_similarity(-1 / 3, .17, .23), .6 * (-1 / 3) + .2 * .17 + .2 * .23)

    def test_invalid_scores_in_each_position(self):
        for value, error in ((None, TypeError), ('1', TypeError), (True, TypeError),
                             (float('nan'), ValueError), (float('inf'), ValueError)):
            for index in range(3):
                scores = [.8, .5, .3]
                scores[index] = value
                with self.subTest(value=value, index=index), self.assertRaises(error):
                    weighted_similarity(*scores)


class HybridScoresTests(unittest.TestCase):
    def setUp(self):
        self.abstracts = {'A': .8, 'B': .2}
        self.titles = {'B': .7, 'A': .5}
        self.citations = {'A': .3, 'B': 1.}

    def test_multiple_candidates_in_different_orders(self):
        self.assertEqual(hybrid_scores(self.abstracts, self.titles, self.citations), {
            'A': .6 * .8 + .2 * .5 + .2 * .3,
            'B': .6 * .2 + .2 * .7 + .2 * 1.,
        })

    def test_custom_weights_iterable(self):
        self.assertEqual(hybrid_scores(self.abstracts, self.titles, self.citations,
                                       weights=iter((0, 1, 0))), self.titles)

    def test_empty(self):
        self.assertEqual(hybrid_scores({}, {}, {}), {})

    def test_invalid_weights_even_when_empty(self):
        with self.assertRaises(ValueError):
            hybrid_scores({}, {}, {}, weights=(0, 0, 0))

    def test_incompatible_ids_in_each_mapping(self):
        for index in range(3):
            for replacement in ({}, {'A': .8}, {'A': .8, 'C': .2}):
                mappings = [self.abstracts, self.titles, self.citations]
                mappings[index] = replacement
                with self.subTest(index=index, replacement=replacement), self.assertRaisesRegex(ValueError, 'same candidate IDs'):
                    hybrid_scores(*mappings)

    def test_no_mutation(self):
        inputs = (self.abstracts, self.titles, self.citations, [.6, .2, .2])
        before = deepcopy(inputs)
        hybrid_scores(*inputs[:3], weights=inputs[3]).clear()
        self.assertEqual(inputs, before)


class RankScoresTests(unittest.TestCase):
    def setUp(self):
        self.scores = {'C': .3, 'A': .9, 'B': .6, 'D': 0.}
        self.expected = [('A', .9), ('B', .6), ('C', .3), ('D', 0.)]

    def test_descending_and_zero(self):
        self.assertEqual(rank_scores(self.scores), self.expected)

    def test_top_k(self):
        for k in (1, 2, 4, 20, None):
            with self.subTest(k=k):
                self.assertEqual(rank_scores(self.scores, top_k=k), self.expected[:k])

    def test_empty(self):
        self.assertEqual(rank_scores({}), [])
        self.assertEqual(rank_scores({}, top_k=1), [])

    def test_invalid_top_k_including_empty(self):
        for scores in ({}, self.scores):
            for k, error in ((0, ValueError), (-1, ValueError), (1.5, TypeError),
                             (1., TypeError), ('1', TypeError), (True, TypeError),
                             (False, TypeError), ([], TypeError)):
                with self.subTest(k=k, scores=scores), self.assertRaises(error):
                    rank_scores(scores, top_k=k)

    def test_tie_break_is_lexicographic_and_order_independent(self):
        items = [('paper_2', .5), ('paper_10', .5), ('paper_1', .5)]
        expected = [('paper_1', .5), ('paper_10', .5), ('paper_2', .5)]
        for scores in (dict(items), dict(reversed(items))):
            self.assertEqual(rank_scores(scores), expected)
            self.assertEqual(rank_scores(scores, top_k=2), expected[:2])

    def test_no_rounding_or_clipping(self):
        scores = {'A': .3, 'B': .3 + 1e-14, 'C': -1 / 3}
        self.assertEqual(rank_scores(scores), [('B', scores['B']), ('A', .3), ('C', -1 / 3)])

    def test_invalid_scores(self):
        for value, error in ((None, TypeError), ('1', TypeError), (True, TypeError),
                             (float('nan'), ValueError), (float('inf'), ValueError)):
            with self.subTest(value=value), self.assertRaises(error):
                rank_scores({'A': value})

    def test_invalid_id(self):
        with self.assertRaises(TypeError):
            rank_scores({1: .5})

    def test_no_mutation(self):
        before = self.scores.copy()
        rank_scores(self.scores, top_k=2).clear()
        self.assertEqual(self.scores, before)


if __name__ == '__main__':
    unittest.main()
