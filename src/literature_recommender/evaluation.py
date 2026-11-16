"""Pure ranking metrics; Liu's hit@k means relevant IDs in top-k divided by k.

The paper's wording is ambiguous; this interpretation follows Table 3 and the
two-reference example. Reproduction policies are in docs/decisions.md.
"""

from collections.abc import Iterable
from numbers import Integral


def _validate_inputs(
    recommended_ids: Iterable[str], relevant_ids: Iterable[str], k: int,
) -> tuple[list[str], set[str]]:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be a positive integer")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    ranking = list(recommended_ids)
    relevant = set(relevant_ids)
    if len(set(ranking)) != len(ranking):
        raise ValueError("Ranking must not contain duplicate IDs, even after top-k")
    if len(ranking) < k:
        raise ValueError("Ranking must contain at least k recommendations")
    if not relevant:
        raise ValueError("Empty ground truth: query is ineligible for evaluation")
    return ranking, relevant


def hit_at_k(
    recommended_ids: Iterable[str], relevant_ids: Iterable[str], k: int,
) -> float:
    """Return |top-k ∩ relevant| / k, retaining the authors' metric name.

    Require positive integer k (not bool), at least k recommendations, globally
    unique ranking IDs and nonempty ground truth. Relevant IDs have set semantics.
    Invalid k types raise TypeError; other violations raise ValueError.
    Collections are not modified; iterators are consumed once.
    """
    ranking, relevant = _validate_inputs(recommended_ids, relevant_ids, k)
    return len(set(ranking[:k]) & relevant) / k


def recall_at_k(
    recommended_ids: Iterable[str], relevant_ids: Iterable[str], k: int,
) -> float:
    """Return |top-k ∩ relevant| / |relevant|, with hit_at_k's validation.

    An empty ground truth raises ValueError, never returns a zero to aggregate.
    """
    ranking, relevant = _validate_inputs(recommended_ids, relevant_ids, k)
    return len(set(ranking[:k]) & relevant) / len(relevant)


def evaluate_ranking(
    recommended_ids: Iterable[str], relevant_ids: Iterable[str],
) -> dict[str, float]:
    """Evaluate hit@1/5/10 and recall@20; require at least 20 recommendations.

    Validate the complete ranking and consume each input iterable only once.
    Ground truth must be defined independently before calling this function.
    """
    ranking, relevant = _validate_inputs(recommended_ids, relevant_ids, 20)
    return {
        "hit@1": hit_at_k(ranking, relevant, 1),
        "hit@5": hit_at_k(ranking, relevant, 5),
        "hit@10": hit_at_k(ranking, relevant, 10),
        "recall@20": recall_at_k(ranking, relevant, 20),
    }
