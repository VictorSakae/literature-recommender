"""Independent similarities; reproduction choices are in docs/decisions.md."""

from collections.abc import Iterable
from math import fsum

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from literature_recommender.graph import get_references


def _unit_embedding(embedding: ArrayLike, name: str) -> np.ndarray:
    """Validate a real finite vector and normalize without changing the input."""
    vector = np.asarray(embedding)
    if vector.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    vector = vector.astype(float)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    scale = np.max(np.abs(vector))
    if scale == 0:
        raise ValueError(f"{name} must have nonzero norm")
    # Scaling first avoids overflow/underflow when squaring finite components.
    vector = vector / scale
    return vector / np.linalg.norm(vector)


def embedding_cosine_similarity(
    embedding_a: ArrayLike, embedding_b: ArrayLike,
) -> float:
    """Return unrounded cosine for two real, finite, nonzero 1-D vectors.

    Reject nonnumeric inputs with TypeError and empty, nonfinite, zero-norm,
    nonvector or mismatched dimensions with ValueError. No API or data access.
    Inputs are not mutated. Signed scores are preserved without clipping;
    floating-point error may cause tiny deviations from [-1, 1].
    """
    a = _unit_embedding(embedding_a, "embedding_a")
    b = _unit_embedding(embedding_b, "embedding_b")
    if a.shape != b.shape:
        raise ValueError("Embedding dimensions must match")
    return float(np.dot(a, b))


def abstract_similarities(
    query_embedding: ArrayLike,
    candidate_embeddings: dict[str, ArrayLike | None],
) -> dict[str, float]:
    """Score supplied candidates, imputing known absent abstracts by their mean.

    None means a known missing abstract, never an embedding or lookup failure.
    The caller maps data-layer abstract absence to None and propagates all other
    errors. Average only valid cosine scores from this call, without combining
    candidate lists. No API calls, rounding, ranking or input mutation.

    A valid query is required even for an empty mapping (which returns {}).
    A missing query is unsupported (TypeError); a nonempty mapping containing
    only None raises ValueError because no mean exists. Invalid vectors raise
    the same errors as embedding_cosine_similarity, never trigger imputation.
    """
    _unit_embedding(query_embedding, "query_embedding")
    if not candidate_embeddings:
        return {}
    scores = {
        paper_id: embedding_cosine_similarity(query_embedding, embedding)
        for paper_id, embedding in candidate_embeddings.items()
        if embedding is not None
    }
    if not scores:
        raise ValueError("No candidate abstract embeddings available for a mean")
    mean = fsum(scores.values()) / len(scores)
    return {
        paper_id: scores[paper_id] if embedding is not None else mean
        for paper_id, embedding in candidate_embeddings.items()
    }


def common_reference_count(
    graph: nx.DiGraph, article_a: str, article_b: str,
) -> int:
    """Count shared outgoing references, including existing self-loop endpoints.

    Common predecessors do not contribute. Raise KeyError for either absent ID.
    Neither the graph nor the reference sets returned by graph queries change.
    """
    references_a = get_references(graph, article_a)
    references_b = get_references(graph, article_b)
    return len(references_a & references_b)


def _minmax_normalize(counts: dict[str, int]) -> dict[str, float]:
    """Normalize supplied counts; empty input stays empty and constants become 0."""
    if not counts:
        return {}
    minimum = min(counts.values())
    maximum = max(counts.values())
    if minimum == maximum:
        return {paper_id: 0.0 for paper_id in counts}
    return {
        paper_id: (count - minimum) / (maximum - minimum)
        for paper_id, count in counts.items()
    }


def citation_similarities(
    graph: nx.DiGraph, query_id: str, candidate_ids: Iterable[str],
) -> dict[str, float]:
    """Return citation scores normalized only across the supplied candidates.

    Reproduction policy: (count - min) / (max - min), without rounding.
    Equal counts (including a single candidate) yield 0.0; no candidates yield
    {}. Zero counts have no special treatment. Lists can be scored separately.

    Raise KeyError for an absent query (even with no candidates) or candidate.
    Inputs are not modified; iterators are consumed once. Repeated IDs produce
    one mapping entry. Candidate order does not affect scores. No ranking or
    candidate filtering is performed, including if query_id is supplied.
    """
    if query_id not in graph:
        raise KeyError(query_id)
    counts = {
        candidate_id: common_reference_count(graph, query_id, candidate_id)
        for candidate_id in candidate_ids
    }
    return _minmax_normalize(counts)


def title_similarity(title_a: str, title_b: str) -> float:
    r"""Return cosine similarity of count BoW vectors over both titles' vocabulary.

    Reproduction policy: lowercase, Unicode tokens matching ``\b\w+\b``
    (including single characters, numbers and underscores), with punctuation
    and hyphens as separators. Preserve accents; no Unicode normalization,
    stopword removal, stemming or lemmatization. Use only word unigrams.

    Raise TypeError for non-str inputs (including None), and ValueError for
    blank titles or titles without tokens. Each title must be valid on its own.
    Return a Python float without rounding; floating-point error can put a
    cosine mathematically equal to one slightly above or below one.
    """
    for name, title in (("title_a", title_a), ("title_b", title_b)):
        if not isinstance(title, str):
            raise TypeError(f"{name} must be a str")
        if not title.strip():
            raise ValueError(f"{name} must not be empty or whitespace-only")

    vectorizer = CountVectorizer(
        input="content",
        lowercase=True,
        strip_accents=None,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        stop_words=None,
        ngram_range=(1, 1),
        binary=False,
        min_df=1,
        max_df=1.0,
        max_features=None,
        vocabulary=None,
    )
    analyze = vectorizer.build_analyzer()
    for name, title in (("title_a", title_a), ("title_b", title_b)):
        if not analyze(title):
            raise ValueError(f"{name} must contain at least one word token")

    counts = vectorizer.fit_transform([title_a, title_b])
    return float(cosine_similarity(counts[0:1], counts[1:2])[0, 0])
