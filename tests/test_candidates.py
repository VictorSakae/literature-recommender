"""Structural candidate selection on small synthetic directed graphs."""

import unittest

import networkx as nx

from literature_recommender.candidates import get_citation_list, get_reference_list


class CandidateTests(unittest.TestCase):
    def test_reference_list_includes_depths_one_two_three_but_not_four(self):
        graph = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        result = get_reference_list(graph, "A")
        self.assertIsInstance(result, set)
        self.assertEqual(result, {"B", "C", "D"})
        self.assertNotIn("E", result)

    def test_reference_list_branches_follow_only_outgoing_edges(self):
        graph = nx.DiGraph([
            ("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"), ("G", "A"),
        ])
        self.assertEqual(get_reference_list(graph, "A"), {"B", "C", "D", "E"})

    def test_multiple_paths_do_not_duplicate_candidates(self):
        graph = nx.DiGraph([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        result = get_reference_list(graph, "A")
        self.assertIsInstance(result, set)
        self.assertEqual(result, {"B", "C", "D"})

    def test_shorter_path_allows_further_expansion_within_limit(self):
        graph = nx.DiGraph([
            ("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"),
            ("D", "E"), ("E", "F"), ("F", "G"),
        ])
        self.assertEqual(get_reference_list(graph, "A"), {"B", "C", "D", "E", "F"})

    def test_citation_list_uses_incoming_neighbors(self):
        graph = nx.DiGraph([("G", "A"), ("H", "A"), ("A", "B")])
        result = get_citation_list(graph, "A")
        self.assertIsInstance(result, set)
        self.assertEqual(result, {"G", "H"})

    def test_citation_list_does_not_expand_predecessors(self):
        graph = nx.DiGraph([("G", "A"), ("H", "A"), ("I", "G"), ("J", "H")])
        self.assertEqual(get_citation_list(graph, "A"), {"G", "H"})

    def test_self_loop_excludes_target_from_both_lists(self):
        graph = nx.DiGraph([("A", "A"), ("A", "B"), ("G", "A")])
        self.assertEqual(get_reference_list(graph, "A"), {"B"})
        self.assertEqual(get_citation_list(graph, "A"), {"G"})

    def test_cycle_terminates_excludes_target_and_respects_depth(self):
        graph = nx.DiGraph([
            ("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E"),
        ])
        self.assertEqual(get_reference_list(graph, "A"), {"B", "C", "D"})
        self.assertEqual(get_citation_list(graph, "A"), {"C"})

    def test_overlap_is_preserved_in_independent_lists(self):
        graph = nx.DiGraph([("A", "B"), ("B", "A")])
        references = get_reference_list(graph, "A")
        citations = get_citation_list(graph, "A")
        self.assertEqual(references, {"B"})
        self.assertEqual(citations, {"B"})
        references.clear()
        self.assertEqual(citations, {"B"})

    def test_existing_nodes_without_neighbors(self):
        graph = nx.DiGraph([("A", "B"), ("X", "X")])
        graph.add_node("isolated")
        self.assertEqual(get_reference_list(graph, "B"), set())
        self.assertEqual(get_citation_list(graph, "A"), set())
        for query in (get_reference_list, get_citation_list):
            for paper_id in ("isolated", "X"):
                with self.subTest(query=query.__name__, paper_id=paper_id):
                    self.assertEqual(query(graph, paper_id), set())

    def test_absent_id_raises_key_error(self):
        for graph in (nx.DiGraph(), nx.DiGraph([("A", "B")])):
            for query in (get_reference_list, get_citation_list):
                with self.subTest(query=query.__name__, nodes=len(graph)):
                    with self.assertRaises(KeyError) as error:
                        query(graph, "missing")
                    self.assertEqual(error.exception.args, ("missing",))

    def test_candidates_do_not_require_metadata(self):
        graph = nx.DiGraph([("A", "external"), ("external", "other"), ("G", "A")])
        graph.nodes["A"]["title"] = "Only target has metadata"
        self.assertEqual(get_reference_list(graph, "A"), {"external", "other"})
        self.assertEqual(get_citation_list(graph, "A"), {"G"})

    def test_queries_and_result_changes_do_not_mutate_graph(self):
        graph = nx.DiGraph([("A", "A"), ("A", "B"), ("B", "C"), ("C", "A")])
        graph.graph["marker"] = "unchanged"
        graph.nodes["A"]["title"] = "Synthetic title"
        graph.edges["A", "B"]["marker"] = "unchanged"
        before = graph.copy()
        for query in (get_reference_list, get_citation_list):
            for paper_id in graph:
                result = query(graph, paper_id)
                result.clear()
                result.add("outside")
            with self.assertRaises(KeyError):
                query(graph, "missing")
        self.assertTrue(nx.utils.graphs_equal(graph, before))


if __name__ == "__main__":
    unittest.main()
