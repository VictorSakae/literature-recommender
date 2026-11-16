"""Title similarity on synthetic strings, independent of metadata and graphs."""

from math import sqrt
import unittest

from literature_recommender.similarity import title_similarity


class TitleSimilarityTests(unittest.TestCase):
    def test_identical_titles(self):
        result = title_similarity("deep learning methods", "deep learning methods")
        self.assertIs(type(result), float)
        self.assertAlmostEqual(result, 1.0, places=14)

    def test_disjoint_titles(self):
        self.assertEqual(title_similarity("graph networks", "bayesian inference"), 0.0)

    def test_partial_overlap(self):
        # Two shared terms, three unit counts per title: 2 / (sqrt(3) * sqrt(3)).
        expected = 2 / 3
        self.assertAlmostEqual(
            title_similarity("deep learning methods", "deep learning applications"),
            expected, places=14,
        )

    def test_term_frequency_changes_similarity(self):
        # In vocabulary [graph, learning], [2, 1] and [1, 1] yield 3 / sqrt(10).
        repeated = title_similarity("graph graph learning", "graph learning")
        self.assertAlmostEqual(repeated, 3 / sqrt(10), places=14)
        self.assertLess(repeated, title_similarity("graph learning", "graph learning"))
        self.assertAlmostEqual(
            title_similarity("graph graph learning", "graph"), 2 / sqrt(5), places=14,
        )

    def test_symmetry(self):
        a, b = "graph graph learning", "learning methods"
        self.assertAlmostEqual(title_similarity(a, b), title_similarity(b, a), places=14)

    def test_word_order_does_not_matter(self):
        self.assertAlmostEqual(title_similarity("deep learning", "learning deep"), 1.0)

    def test_lowercase_punctuation_and_hyphens(self):
        self.assertAlmostEqual(
            title_similarity("Deep-learning, METHODS!", "deep learning methods"), 1.0,
        )

    def test_single_character_tokens_and_numbers_are_retained(self):
        self.assertAlmostEqual(title_similarity("A x 2", "x"), 1 / sqrt(3), places=14)
        self.assertAlmostEqual(title_similarity("x", "X"), 1.0)

    def test_underscores_are_word_characters(self):
        self.assertEqual(title_similarity("deep_learning", "deep learning"), 0.0)
        self.assertAlmostEqual(title_similarity("_", "_"), 1.0)

    def test_accents_are_preserved_without_unicode_normalization(self):
        self.assertAlmostEqual(title_similarity("CAFÉ", "café"), 1.0)
        self.assertEqual(title_similarity("café", "cafe"), 0.0)
        self.assertEqual(title_similarity("café", "cafe\u0301"), 0.0)

    def test_stopwords_are_retained(self):
        self.assertAlmostEqual(title_similarity("the graph", "graph"), 1 / sqrt(2))

    def test_no_stemming_or_lemmatization(self):
        self.assertEqual(title_similarity("learning methods", "learn method"), 0.0)

    def test_non_string_inputs_raise_type_error_in_either_position(self):
        for invalid in (None, 42, 3.5, False, ["graph"], {"title": "graph"}, b"graph"):
            for args, name in (((invalid, "graph"), "title_a"),
                               (("graph", invalid), "title_b")):
                with self.subTest(args=args):
                    with self.assertRaisesRegex(TypeError, name):
                        title_similarity(*args)

    def test_blank_and_tokenless_titles_raise_value_error_in_either_position(self):
        for invalid in ("", " \t\n", "--- ... !", "∑ + 😀", "\u0301"):
            for args, name in (((invalid, "graph"), "title_a"),
                               (("graph", invalid), "title_b"),
                               ((invalid, invalid), "title_a")):
                with self.subTest(args=args):
                    with self.assertRaisesRegex(ValueError, name):
                        title_similarity(*args)


if __name__ == "__main__":
    unittest.main()
