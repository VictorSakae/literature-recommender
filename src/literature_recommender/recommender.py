"""Combine existing similarities and rank one supplied candidate list at a time."""

from collections.abc import Iterable, Mapping
from math import fsum, isclose, isfinite
from numbers import Integral, Real

import networkx as nx
import pandas as pd
from numpy.typing import ArrayLike

from literature_recommender.data import get_article
from literature_recommender.similarity import (
    abstract_similarities,
    citation_similarities,
    title_similarity,
)


def _finite_score(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_weights(weights: Iterable[float]) -> tuple[float, ...]:
    weights = tuple(weights)
    if len(weights) != 3:
        raise ValueError("Exactly three weights are required")
    weights = tuple(_finite_score(value, "weight") for value in weights)
    if any(value < 0 for value in weights):
        raise ValueError("Weights must be nonnegative")
    if not isclose(fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Weights must sum to 1 (absolute tolerance 1e-9)")
    return weights


def weighted_similarity(
    abstract_similarity: float,
    title_similarity: float,
    citation_similarity: float,
    *,
    weights: Iterable[float] = (0.6, 0.2, 0.2),
) -> float:
    """Return the weighted sum in abstract/title/citation order, without rounding.

    Weights must be three finite real nonnegative numbers summing to 1 within
    absolute tolerance 1e-9. Scores must be finite real numbers; signed values
    are preserved. Booleans are rejected. No normalization or data access.
    """
    weights = _validate_weights(weights)
    scores = (
        _finite_score(abstract_similarity, "abstract_similarity"),
        _finite_score(title_similarity, "title_similarity"),
        _finite_score(citation_similarity, "citation_similarity"),
    )
    return sum(weight * score for weight, score in zip(weights, scores))


def hybrid_scores(
    abstract_scores: Mapping[str, float],
    title_scores: Mapping[str, float],
    citation_scores: Mapping[str, float],
    *,
    weights: Iterable[float] = (0.6, 0.2, 0.2),
) -> dict[str, float]:
    """Combine already computed scores with exactly matching IDs, without mutation.

    Mismatched ID sets raise ValueError, including a partially empty input.
    All-empty mappings return {} after validating weights. No ranking is done.
    """
    weights = _validate_weights(weights)
    if not (set(abstract_scores) == set(title_scores) == set(citation_scores)):
        raise ValueError("Similarity mappings must contain exactly the same candidate IDs")
    return {
        paper_id: weighted_similarity(
            abstract_scores[paper_id], title_scores[paper_id], citation_scores[paper_id],
            weights=weights,
        )
        for paper_id in abstract_scores
    }


def _validate_top_k(top_k: int | None) -> None:
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, Integral):
            raise TypeError("top_k must be a positive integer or None")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer or None")


def rank_scores(
    scores: Mapping[str, float], *, top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Return (paper_id, score) pairs by descending score, then ascending string ID.

    The lexicographic tie-break is a reproduction choice. None returns all;
    a positive integer limits the result. Validate even for empty mappings.
    Scores are finite real numbers, preserved without rounding or mutation.
    """
    _validate_top_k(top_k)
    for paper_id, score in scores.items():
        if not isinstance(paper_id, str):
            raise TypeError("paper_id must be a string")
        _finite_score(score, "score")
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def rank_candidate_list(
    papers: pd.DataFrame,
    graph: nx.DiGraph,
    query_id: str,
    candidate_ids: Iterable[str],
    embeddings: Mapping[str, ArrayLike | None],
    *,
    weights: Iterable[float] = (0.6, 0.2, 0.2),
    top_k: int | None = None,
) -> dict:
    """Rank one already selected list using normalized metadata and supplied vectors.

    Return a dict with:
    - ranking: ordered row dicts containing paper_id, abstract_similarity,
      title_similarity, citation_similarity and weighted_similarity (up to top_k).
    - eligible_ids: all eligible IDs in lexicographic order, before top_k.
    - excluded: {paper_id: "missing_metadata"} for IDs absent from papers.

    Exclude absent metadata before any similarity calculation. A metadata row
    with pd.NA abstract remains eligible and must have an explicit None entry
    in embeddings. Every other eligible candidate and the query require valid
    vectors. Missing keys and inconsistent absence markers raise errors; no
    technical failure is imputed. Extra embeddings, including excluded IDs,
    are unused. A query with missing abstract is unsupported.

    Reuse the existing per-call normalization and imputation over all eligible
    candidates, then truncate the ranking. Even an empty list requires a query
    with metadata, a graph node and a valid embedding. Repeated candidate IDs
    collapse. No structural selection, list merging, API calls or input mutation.
    """
    weights = _validate_weights(weights)
    _validate_top_k(top_k)
    query = get_article(papers, query_id)
    if pd.isna(query['abstract']):
        raise ValueError("Query must have an available abstract")

    candidate_ids = set(candidate_ids)
    if any(not isinstance(paper_id, str) for paper_id in candidate_ids):
        raise TypeError("paper_id must be a string")
    metadata_ids = set(papers['paper_id'])
    eligible_ids = sorted(candidate_ids & metadata_ids)
    excluded = {
        paper_id: 'missing_metadata'
        for paper_id in sorted(candidate_ids - metadata_ids)
    }
    articles = {paper_id: get_article(papers, paper_id) for paper_id in eligible_ids}

    title_scores = {
        paper_id: title_similarity(query['title'], article['title'])
        for paper_id, article in articles.items()
    }
    citation_scores = citation_similarities(graph, query_id, eligible_ids)
    candidate_embeddings = {}
    for paper_id, article in articles.items():
        embedding = embeddings[paper_id]
        if pd.isna(article['abstract']) != (embedding is None):
            raise ValueError(
                f"Embedding for {paper_id!r} must be None exactly when its abstract is missing"
            )
        candidate_embeddings[paper_id] = embedding
    abstract_scores = abstract_similarities(embeddings[query_id], candidate_embeddings)
    scores = hybrid_scores(abstract_scores, title_scores, citation_scores, weights=weights)
    ranking = [
        {
            'paper_id': paper_id,
            'abstract_similarity': abstract_scores[paper_id],
            'title_similarity': title_scores[paper_id],
            'citation_similarity': citation_scores[paper_id],
            'weighted_similarity': score,
        }
        for paper_id, score in rank_scores(scores, top_k=top_k)
    ]
    return {'ranking': ranking, 'eligible_ids': eligible_ids, 'excluded': excluded}
