"""Citation counts and normalization on small synthetic directed graphs."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import networkx as nx

from literature_recommender.similarity import (
    citation_similarities,
    common_reference_count,
)


class CommonReferenceCountTests(unittest.TestCase):
    def test_one_shared_reference(self):
        graph = nx.DiGraph([
            ("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Z"),
        ])
        count = common_reference_count(graph, "A", "B")
        self.assertIs(type(count), int)
        self.assertEqual(count, 1)

    def test_multiple_shared_references_and_symmetry(self):
        graph = nx.DiGraph([
            ("A", "X"), ("A", "Y"), ("A", "Z"),
            ("B", "X"), ("B", "Y"), ("B", "W"),
        ])
        self.assertEqual(common_reference_count(graph, "A", "B"), 2)
        self.assertEqual(common_reference_count(graph, "B", "A"), 2)

    def test_disjoint_references(self):
        graph = nx.DiGraph([("A", "X"), ("B", "Y")])
        self.assertEqual(common_reference_count(graph, "A", "B"), 0)

    def test_common_predecessor_does_not_count(self):
        graph = nx.DiGraph([("X", "A"), ("X", "B")])
        self.assertEqual(common_reference_count(graph, "A", "B"), 0)

    def test_missing_id_in_either_position(self):
        graph = nx.DiGraph([("A", "X")])
        for pair in (("missing", "A"), ("A", "missing")):
            with self.subTest(pair=pair):
                with self.assertRaises(KeyError) as error:
                    common_reference_count(graph, *pair)
                self.assertEqual(error.exception.args, ("missing",))

    def test_self_loops_at_either_endpoint_contribute(self):
        graph = nx.DiGraph([
            ("A", "A"), ("B", "A"), ("A", "B"), ("B", "B"),
        ])
        # Both A and B are shared references, despite being the compared nodes.
        self.assertEqual(common_reference_count(graph, "A", "B"), 2)
        self.assertEqual(common_reference_count(graph, "B", "A"), 2)

    def test_same_article_counts_its_references(self):
        graph = nx.DiGraph([("A", "A"), ("A", "X")])
        self.assertEqual(common_reference_count(graph, "A", "A"), 2)

    def test_reference_sets_are_not_modified(self):
        references_a, references_b = {"X", "Y"}, {"X", "Z"}
        with patch("literature_recommender.similarity.get_references",
                   side_effect=[references_a, references_b]):
            self.assertEqual(common_reference_count(nx.DiGraph(), "A", "B"), 1)
        self.assertEqual(references_a, {"X", "Y"})
        self.assertEqual(references_b, {"X", "Z"})


class CitationSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.DiGraph([
            ("Q", "W"), ("Q", "X"), ("Q", "Y"), ("Q", "Z"),
            ("B", "W"), ("B", "X"),
            ("C", "W"), ("C", "X"), ("C", "Y"), ("C", "Z"),
            ("D", "W"), ("D", "X"), ("D", "Y"),
        ])
        self.graph.add_node("A")

    def test_known_normalization(self):
        counts = {p: common_reference_count(self.graph, "Q", p) for p in "ABC"}
        self.assertEqual(counts, {"A": 0, "B": 2, "C": 4})
        scores = citation_similarities(self.graph, "Q", ["A", "B", "C"])
        self.assertEqual(scores, {"A": 0.0, "B": 0.5, "C": 1.0})
        self.assertTrue(all(type(score) is float for score in scores.values()))

    def test_extrema_use_only_supplied_candidates_with_positive_minimum(self):
        # Counts 2, 3, 4: the graph's zero-count nodes must not set the minimum.
        self.assertEqual(
            citation_similarities(self.graph, "Q", ["B", "D", "C"]),
            {"B": 0.0, "D": 0.5, "C": 1.0},
        )
        self.assertEqual(
            citation_similarities(self.graph, "Q", ["B", "D"]),
            {"B": 0.0, "D": 1.0},
        )

    def test_no_internal_rounding(self):
        self.graph.add_edge("E", "W")
        self.assertEqual(
            citation_similarities(self.graph, "Q", ["A", "E", "D"]),
            {"A": 0.0, "E": 1 / 3, "D": 1.0},
        )

    def test_all_counts_zero(self):
        self.assertEqual(
            citation_similarities(self.graph, "Q", ["A", "W", "X"]),
            {"A": 0.0, "W": 0.0, "X": 0.0},
        )

    def test_all_counts_equal_and_positive(self):
        self.graph.add_edges_from([
            ("E", "W"), ("E", "X"), ("F", "Y"), ("F", "Z"),
        ])
        for candidate in ("B", "E", "F"):
            self.assertEqual(common_reference_count(self.graph, "Q", candidate), 2)
        self.assertEqual(
            citation_similarities(self.graph, "Q", ["B", "E", "F"]),
            {"B": 0.0, "E": 0.0, "F": 0.0},
        )

    def test_single_candidate(self):
        for candidate in ("A", "B"):
            with self.subTest(candidate=candidate):
                self.assertEqual(
                    citation_similarities(self.graph, "Q", [candidate]),
                    {candidate: 0.0},
                )

    def test_empty_candidates(self):
        self.assertEqual(citation_similarities(self.graph, "Q", set()), {})

    def test_missing_query_including_empty_candidates(self):
        for candidates in (["A"], set()):
            with self.subTest(candidates=candidates):
                with self.assertRaises(KeyError) as error:
                    citation_similarities(self.graph, "missing", candidates)
                self.assertEqual(error.exception.args, ("missing",))

    def test_missing_candidate(self):
        with self.assertRaises(KeyError) as error:
            citation_similarities(self.graph, "Q", ["A", "missing"])
        self.assertEqual(error.exception.args, ("missing",))

    def test_self_loop_affects_normalized_score(self):
        graph = nx.DiGraph([("Q", "Q"), ("B", "Q")])
        graph.add_node("A")
        self.assertEqual(common_reference_count(graph, "Q", "B"), 1)
        self.assertEqual(
            citation_similarities(graph, "Q", ["A", "B"]), {"A": 0.0, "B": 1.0},
        )

    def test_candidate_order_does_not_affect_scores(self):
        expected = {"A": 0.0, "B": 0.5, "C": 1.0}
        for candidates in (["A", "B", "C"], ["C", "A", "B"], {"B", "C", "A"}):
            with self.subTest(candidates=candidates):
                self.assertEqual(citation_similarities(self.graph, "Q", candidates), expected)

    def test_generator_and_repeated_ids(self):
        candidates = (p for p in ["C", "B", "A", "B"])
        self.assertEqual(
            citation_similarities(self.graph, "Q", candidates),
            {"A": 0.0, "B": 0.5, "C": 1.0},
        )

    def test_inputs_are_not_mutated(self):
        self.graph.graph["marker"] = ["unchanged"]
        self.graph.nodes["Q"]["marker"] = ["unchanged"]
        self.graph.edges["Q", "W"]["marker"] = ["unchanged"]
        before_graph = deepcopy(self.graph)
        for candidates in (["C", "A", "B", "B"], {"C", "A", "B"}):
            before_candidates = candidates.copy()
            common_reference_count(self.graph, "Q", "B")
            scores = citation_similarities(self.graph, "Q", candidates)
            scores.clear()
            self.assertEqual(candidates, before_candidates)
        with self.assertRaises(KeyError):
            citation_similarities(self.graph, "Q", ["B", "missing"])
        self.assertTrue(nx.utils.graphs_equal(self.graph, before_graph))


if __name__ == "__main__":
    unittest.main()
