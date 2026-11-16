"""Basic citation graph behavior using only small synthetic relations."""

import unittest

import networkx as nx
import pandas as pd
from pandas.testing import assert_frame_equal

from literature_recommender.graph import (
    build_citation_graph,
    get_citations,
    get_references,
)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.edges = [
            ("A", "B"), ("A", "C"), ("B", "D"), ("C", "E"),
            ("G", "A"), ("H", "A"), ("X", "X"),
        ]
        self.citations = pd.DataFrame(
            self.edges, columns=["source_id", "target_id"], dtype="string"
        )
        self.graph = build_citation_graph(self.citations)

    def test_directed_graph_preserves_exact_relations(self):
        self.assertIs(type(self.graph), nx.DiGraph)
        self.assertEqual(set(self.graph.edges), set(self.edges))
        self.assertFalse(self.graph.has_edge("B", "A"))

    def test_references_are_successors(self):
        references = get_references(self.graph, "A")
        self.assertIsInstance(references, set)
        self.assertEqual(references, {"B", "C"})

    def test_citations_are_predecessors(self):
        citations = get_citations(self.graph, "A")
        self.assertIsInstance(citations, set)
        self.assertEqual(citations, {"G", "H"})

    def test_existing_nodes_without_neighbors(self):
        self.assertEqual(get_references(self.graph, "D"), set())
        self.assertEqual(get_citations(self.graph, "G"), set())

    def test_absent_id_raises_key_error(self):
        for query in (get_references, get_citations):
            with self.subTest(query=query.__name__):
                with self.assertRaises(KeyError) as error:
                    query(self.graph, "missing")
                self.assertEqual(error.exception.args, ("missing",))

    def test_duplicate_edges_collapse_but_reverse_pair_is_distinct(self):
        citations = pd.DataFrame(
            [("A", "B"), ("A", "B"), ("B", "A")],
            columns=["source_id", "target_id"], dtype="string",
        )
        before = citations.copy(deep=True)
        graph = build_citation_graph(citations)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(set(graph.edges), {("A", "B"), ("B", "A")})
        assert_frame_equal(citations, before)

    def test_self_loop_is_preserved_in_both_queries(self):
        self.assertTrue(self.graph.has_edge("X", "X"))
        self.assertEqual(nx.number_of_selfloops(self.graph), 1)
        self.assertEqual(get_references(self.graph, "X"), {"X"})
        self.assertEqual(get_citations(self.graph, "X"), {"X"})

    def test_nodes_are_only_endpoints_without_metadata(self):
        self.assertEqual(set(self.graph), {"A", "B", "C", "D", "E", "G", "H", "X"})
        self.assertTrue(all(attributes == {} for _, attributes in self.graph.nodes(data=True)))

    def test_construction_and_queries_do_not_mutate_inputs(self):
        before = self.citations.copy(deep=True)
        graph = build_citation_graph(self.citations)
        graph.nodes["A"]["title"] = "Synthetic title"
        graph.edges["A", "B"]["marker"] = "unchanged"
        before_graph = graph.copy()
        for query in (get_references, get_citations):
            for paper_id in graph:
                result = query(graph, paper_id)
                result.clear()
                result.add("outside")
            with self.assertRaises(KeyError):
                query(graph, "missing")
        assert_frame_equal(self.citations, before)
        self.assertTrue(nx.utils.graphs_equal(graph, before_graph))

    def test_empty_citations_create_empty_graph(self):
        graph = build_citation_graph(self.citations.iloc[:0])
        self.assertIs(type(graph), nx.DiGraph)
        self.assertEqual(graph.number_of_nodes(), 0)
        self.assertEqual(graph.number_of_edges(), 0)
        for query in (get_references, get_citations):
            with self.assertRaises(KeyError):
                query(graph, "missing")


if __name__ == "__main__":
    unittest.main()
