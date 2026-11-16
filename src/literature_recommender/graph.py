"""Represent citations as directed edges from citing to cited article."""

import networkx as nx
import pandas as pd


def build_citation_graph(citations: pd.DataFrame) -> nx.DiGraph:
    """Build a graph from normalized source_id/target_id string columns.

    Only citation endpoints become nodes; no metadata is required. Duplicate
    directed pairs become one edge and self-loops are preserved. The input
    DataFrame is not modified; normalization belongs to the data layer.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(
        citations[["source_id", "target_id"]].itertuples(index=False, name=None)
    )
    return graph


def get_references(graph: nx.DiGraph, paper_id: str) -> set[str]:
    """Return articles cited by paper_id; raise KeyError if the ID is absent."""
    if paper_id not in graph:
        raise KeyError(paper_id)
    return set(graph.successors(paper_id))


def get_citations(graph: nx.DiGraph, paper_id: str) -> set[str]:
    """Return articles citing paper_id; raise KeyError if the ID is absent."""
    if paper_id not in graph:
        raise KeyError(paper_id)
    return set(graph.predecessors(paper_id))
