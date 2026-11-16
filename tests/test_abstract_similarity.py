"""Pure vector and missing-abstract policies, using only synthetic data."""

from copy import deepcopy
from math import sqrt
import unittest

import numpy as np

from literature_recommender.similarity import (
    abstract_similarities,
    embedding_cosine_similarity as cosine,
)


class EmbeddingCosineTests(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertIs(type(cosine([1, 0], [1, 0])), float)

    def test_orthogonal(self):
        self.assertEqual(cosine([1, 0], [0, 1]), 0.0)

    def test_intermediate_without_rounding(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 2]), 1 / sqrt(5), places=15)

    def test_negative(self):
        self.assertEqual(cosine([1, 0], [-1, 0]), -1.0)

    def test_symmetry(self):
        self.assertEqual(cosine([2, 3], [4, 1]), cosine([4, 1], [2, 3]))

    def test_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            cosine([1, 0], [1, 0, 0])

    def test_empty_nonvector_zero_and_nonfinite_in_either_position(self):
        for invalid in ([], [[1, 0]], 1, [0, 0], [np.nan, 1], [np.inf, 1]):
            for pair in ((invalid, [1, 0]), ([1, 0], invalid)):
                with self.subTest(pair=pair), self.assertRaises(ValueError):
                    cosine(*pair)

    def test_nonnumeric_in_either_position(self):
        for invalid in (None, "text", ["1", "0"], [1j, 0], [True, False], {}):
            for pair in ((invalid, [1, 0]), ([1, 0], invalid)):
                with self.subTest(pair=pair), self.assertRaises(TypeError):
                    cosine(*pair)

    def test_ragged_vector(self):
        with self.assertRaises(ValueError):
            cosine([[1], [1, 2]], [1, 0])

    def test_scale_invariance_for_extreme_finite_values(self):
        for scale in (1e-300, 1e300):
            with self.subTest(scale=scale):
                self.assertAlmostEqual(cosine([scale, scale], [1, 0]), 1 / sqrt(2))

    def test_no_mutation(self):
        a, b = np.array([3., 4.]), [1, 2]
        before_a, before_b = a.copy(), b.copy()
        a.flags.writeable = False
        cosine(a, b)
        np.testing.assert_array_equal(a, before_a)
        self.assertEqual(b, before_b)


class AbstractSimilaritiesTests(unittest.TestCase):
    def setUp(self):
        self.query = [1., 0.]
        self.a = [0.4, sqrt(0.84)]
        self.c = [0.8, 0.6]

    def test_all_available(self):
        scores = abstract_similarities(self.query, {"A": self.a, "C": self.c})
        self.assertAlmostEqual(scores["A"], 0.4)
        self.assertAlmostEqual(scores["C"], 0.8)
        self.assertTrue(all(type(score) is float for score in scores.values()))

    def test_one_missing_manual_mean(self):
        scores = abstract_similarities(self.query, {"A": self.a, "B": None, "C": self.c})
        self.assertAlmostEqual(scores["A"], 0.4)
        self.assertAlmostEqual(scores["B"], 0.6)
        self.assertAlmostEqual(scores["C"], 0.8)

    def test_several_missing_do_not_enter_denominator(self):
        scores = abstract_similarities(
            self.query, {"B": None, "A": self.a, "D": None, "C": self.c},
        )
        self.assertAlmostEqual(scores["B"], 0.6)
        self.assertEqual(scores["B"], scores["D"])
        self.assertEqual(list(scores), ["B", "A", "D", "C"])

    def test_mean_is_local_to_each_call(self):
        reference = abstract_similarities(self.query, {"A": self.a, "B": None})
        citation = abstract_similarities(self.query, {"C": self.c, "B": None})
        self.assertAlmostEqual(reference["B"], 0.4)
        self.assertAlmostEqual(citation["B"], 0.8)

    def test_zero_and_negative_scores_are_valid(self):
        self.assertEqual(
            abstract_similarities(self.query, {"A": [-1, 0], "B": None, "C": [0, 1]}),
            {"A": -1., "B": -0.5, "C": 0.},
        )

    def test_empty_candidates(self):
        self.assertEqual(abstract_similarities(self.query, {}), {})

    def test_all_missing(self):
        for candidates in ({"A": None}, {"A": None, "B": None}):
            with self.subTest(candidates=candidates), self.assertRaisesRegex(ValueError, "No candidate"):
                abstract_similarities(self.query, candidates)

    def test_query_is_required_even_without_candidates(self):
        for candidates in ({}, {"A": self.a}, {"A": None}):
            with self.subTest(candidates=candidates), self.assertRaises(TypeError):
                abstract_similarities(None, candidates)

    def test_invalid_query_even_without_candidates(self):
        with self.assertRaises(ValueError):
            abstract_similarities([0, 0], {})

    def test_invalid_candidates_are_not_imputed(self):
        for invalid in ([], [0, 0], [1, 2, 3], [np.nan, 0], "bad", ["1", "0"]):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                abstract_similarities(self.query, {"A": self.a, "B": invalid, "C": None})

    def test_candidate_order_does_not_change_scores(self):
        candidates = {"A": self.a, "B": None, "C": self.c}
        self.assertEqual(
            abstract_similarities(self.query, candidates),
            abstract_similarities(self.query, dict(reversed(list(candidates.items())))),
        )

    def test_no_mutation(self):
        candidates = {"A": self.a, "B": None, "C": self.c}
        before = deepcopy((self.query, candidates))
        abstract_similarities(self.query, candidates).clear()
        self.assertEqual((self.query, candidates), before)


if __name__ == "__main__":
    unittest.main()
