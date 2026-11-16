"""Select structural candidates using Liu et al.'s two independent lists."""

import networkx as nx

from literature_recommender.graph import get_citations, get_references


def get_reference_list(graph: nx.DiGraph, paper_id: str) -> set[str]:
    """Return nodes reached by 1–3 outgoing edges, excluding paper_id.

    Expand by levels, visiting each node once to handle cycles and shared paths.
    No metadata is required and the graph is not modified. An absent paper_id
    raises KeyError through the graph layer.
    """
    references: set[str] = set()
    visited = {paper_id}
    frontier = {paper_id}
    for _ in range(3):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(get_references(graph, node))
        next_frontier.difference_update(visited)
        references.update(next_frontier)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    references.discard(paper_id)
    return references


def get_citation_list(graph: nx.DiGraph, paper_id: str) -> set[str]:
    """Return direct incoming neighbors, excluding paper_id even for self-loops.

    No metadata is required and the graph is not modified. An absent paper_id
    raises KeyError through the graph layer. Overlap with REFERENCE LIST stays.
    """
    citations = get_citations(graph, paper_id)
    citations.discard(paper_id)
    return citations
