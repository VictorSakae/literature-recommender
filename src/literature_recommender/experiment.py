"""Prepare a small reference-reconstruction experiment without computing scores.

Metadata must follow load_papers' contract. No provider calls or embeddings are
created here. Ground truth is independent of candidates and recommendation scores.
"""

from collections.abc import Set
from random import Random

import networkx as nx
import pandas as pd

from literature_recommender.candidates import get_reference_list
from literature_recommender.graph import get_references


def available_references(
    graph: nx.DiGraph, query_id: str, metadata_ids: Set[str],
) -> set[str]:
    """Direct outgoing references with metadata, excluding the query itself.

    Abstract absence does not remove relevance. Empty results mark ineligible
    queries; the caller must not turn them into zero-valued evaluation rows.
    """
    return (get_references(graph, query_id) & metadata_ids) - {query_id}


def _abstract_lookup(papers: pd.DataFrame) -> dict[str, str | None]:
    if papers['paper_id'].duplicated().any():
        raise ValueError("Duplicate article IDs make experiment metadata ambiguous")
    return {
        paper_id: None if pd.isna(abstract) else abstract
        for paper_id, abstract in papers[['paper_id', 'abstract']].itertuples(index=False, name=None)
    }


def reference_query_population(papers: pd.DataFrame, graph: nx.DiGraph) -> pd.DataFrame:
    """Enumerate all eligible queries, before any candidate-pool cost limit.

    Require metadata, a graph node, query abstract, nonempty available references
    and >=20 metadata-eligible REFERENCE LIST candidates. Also require at least
    one candidate abstract, since otherwise stage 3C cannot impute similarities.
    Return counts only; never retain full candidate pools for the whole corpus.
    Invalid titles/embeddings remain errors at ranking time, not silent exclusions.
    """
    abstracts = _abstract_lookup(papers)
    metadata_ids = set(abstracts)
    columns = ['query_id', 'n_ground_truth', 'n_structural_candidates',
               'n_eligible_candidates', 'n_candidates_without_abstract',
               'n_candidates_without_metadata']
    rows = []
    for query_id in sorted(metadata_ids & graph.nodes):
        if abstracts[query_id] is None:
            continue
        relevant = available_references(graph, query_id, metadata_ids)
        if not relevant:
            continue
        candidates = get_reference_list(graph, query_id)
        eligible = candidates & metadata_ids
        if len(eligible) < 20:
            continue
        missing = sum(abstracts[paper_id] is None for paper_id in eligible)
        if missing == len(eligible):
            continue
        rows.append([query_id, len(relevant), len(candidates), len(eligible),
                     missing, len(candidates) - len(eligible)])
    return pd.DataFrame(rows, columns=columns)


def select_query_sample(
    population: pd.DataFrame, *, seed: int, max_queries: int = 5,
    max_candidates: int | None = None,
) -> list[str]:
    """Sample sorted eligible IDs without replacement using a local random state.

    A cost cap restricts queries before sampling, never their candidate lists.
    At most five queries; if fewer are eligible, return them all in ID order.
    """
    if type(max_queries) is not int or not 1 <= max_queries <= 5:
        raise ValueError("max_queries must be an integer between 1 and 5")
    if max_candidates is not None and (type(max_candidates) is not int or max_candidates < 20):
        raise ValueError("max_candidates must be an integer >=20 or None")
    if population['query_id'].duplicated().any():
        raise ValueError("Population must contain unique query IDs")
    allowed = population[population['n_eligible_candidates'] >= 20]
    if max_candidates is not None:
        allowed = allowed[allowed['n_eligible_candidates'] <= max_candidates]
    ids = sorted(allowed['query_id'])
    return ids if len(ids) <= max_queries else Random(seed).sample(ids, max_queries)


def prepare_reference_query(
    papers: pd.DataFrame, graph: nx.DiGraph, query_id: str,
) -> dict:
    """Prepare full REFERENCE LIST, independent truth, texts and readiness counts.

    No candidate truncation. Exclusions use stage 4's missing_metadata reason.
    Abstracts map query and every eligible ID to its text or known absence None.
    Count unique nonmissing text values (including the query), not unique IDs.
    """
    abstracts = _abstract_lookup(papers)
    if abstracts[query_id] is None:
        raise ValueError("Query must have an available abstract")
    relevant = available_references(graph, query_id, abstracts.keys())
    if not relevant:
        raise ValueError("Empty ground truth: query is ineligible for evaluation")
    candidates = get_reference_list(graph, query_id)
    eligible = candidates & abstracts.keys()
    if len(eligible) < 20:
        raise ValueError("Query must have at least 20 eligible candidates")
    missing = sum(abstracts[paper_id] is None for paper_id in eligible)
    if missing == len(eligible):
        raise ValueError("No candidate abstracts available for imputation")
    texts = {paper_id: abstracts[paper_id] for paper_id in sorted(eligible | {query_id})}
    excluded = {paper_id: 'missing_metadata' for paper_id in sorted(candidates - eligible)}
    return {
        'query_id': query_id,
        'relevant_ids': relevant,
        'candidate_ids': candidates,
        'eligible_ids': eligible,
        'excluded': excluded,
        'abstracts': texts,
        'diagnostic': {
            'query_id': query_id,
            'n_ground_truth': len(relevant),
            'n_structural_candidates': len(candidates),
            'n_eligible_candidates': len(eligible),
            'n_candidates_without_abstract': missing,
            'n_candidates_without_metadata': len(excluded),
            'n_unique_abstracts_needed': len({text for text in texts.values() if text is not None}),
        },
    }


ELIGIBILITY_RULES = [
    'metadata available', 'query exists in graph', 'query abstract available',
    'nonempty available direct-reference ground truth',
    'at least candidate_min metadata-eligible REFERENCE LIST candidates',
    'query has valid fixed textual representations',
    'at least one candidate abstract permits a real textual comparison',
    'all available candidate abstracts have valid fixed textual representations',
]


def build_experiment_population(
    papers: pd.DataFrame, graph: nx.DiGraph, *,
    textual_valid_ids: Set[str], candidate_min: int = 20,
) -> pd.DataFrame:
    """Audit metadata IDs UNION graph IDs, without scores or candidate truncation.

    papers follows load_papers' normalized contract. textual_valid_ids must be
    checked against the fixed representations before calling (no model fitting).
    Invalid available candidate text is a technical incompatibility, never an
    imputable absence. Counts are retained even for excluded queries. The first
    failed rule supplies the exclusive exclusion reason; no ID is silently lost.
    """
    if type(candidate_min) is not int or candidate_min < 20:
        raise ValueError('candidate_min must be an integer >=20')
    abstracts = _abstract_lookup(papers)
    metadata = set(abstracts)
    rows = []
    for q in sorted(metadata | set(graph)):
        candidates = get_reference_list(graph, q) if q in graph else set()
        eligible = candidates & metadata
        truth = available_references(graph, q, metadata) if q in graph else set()
        with_text = {p for p in eligible if abstracts[p] is not None}
        reason = ''
        if q not in metadata:
            reason = 'NO_METADATA'
        elif q not in graph:
            reason = 'NOT_IN_GRAPH'
        elif abstracts[q] is None:
            reason = 'NO_QUERY_ABSTRACT'
        elif not truth:
            reason = 'EMPTY_GROUND_TRUTH'
        elif len(eligible) < candidate_min:
            reason = 'TOO_FEW_ELIGIBLE_CANDIDATES'
        elif q not in textual_valid_ids or not (with_text & textual_valid_ids):
            reason = 'NO_VALID_TEXTUAL_COMPARISON'
        elif not with_text <= textual_valid_ids:
            reason = 'INVALID_CANDIDATE_REPRESENTATION'
        rows.append(dict(
            query_id=q, eligible=not reason, exclusion_reason=reason,
            n_ground_truth=len(truth), n_structural_candidates=len(candidates),
            n_eligible_candidates=len(eligible), n_candidates_with_abstract=len(with_text),
            n_candidates_without_abstract=len(eligible) - len(with_text),
            missing_abstract_fraction=(len(eligible) - len(with_text)) / len(eligible)
            if eligible else 0.0,
        ))
    return pd.DataFrame(rows, columns=[
        'query_id', 'eligible', 'exclusion_reason', 'n_ground_truth',
        'n_structural_candidates', 'n_eligible_candidates',
        'n_candidates_with_abstract', 'n_candidates_without_abstract',
        'missing_abstract_fraction',
    ])


def split_experiment_queries(
    population: pd.DataFrame, *, n_tuning: int, n_evaluation: int,
    seed: int = 42, candidate_min: int = 20, candidate_max: int | None = None,
) -> dict:
    """Sample tuning then evaluation from sorted remaining IDs, without replacement.

    Uses a local Python Random and does not mutate inputs. Bounds filter queries,
    never candidates. A different seed or size is a new protocol, not a retry.
    """
    for name, value in [('n_tuning', n_tuning), ('n_evaluation', n_evaluation)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    if type(candidate_min) is not int or candidate_min < 20:
        raise ValueError('candidate_min must be an integer >=20')
    if candidate_max is not None and (type(candidate_max) is not int or candidate_max < candidate_min):
        raise ValueError('candidate_max must be None or an integer >= candidate_min')
    if population.query_id.duplicated().any():
        raise ValueError('Population must contain unique query IDs')
    allowed = population.loc[population.eligible &
                             (population.n_eligible_candidates >= candidate_min)]
    if candidate_max is not None:
        allowed = allowed.loc[allowed.n_eligible_candidates <= candidate_max]
    ids = sorted(allowed.query_id)
    if len(ids) < n_tuning + n_evaluation:
        raise ValueError('Insufficient eligible population for requested split sizes')
    rng = Random(seed)
    tuning = rng.sample(ids, n_tuning)
    selected = set(tuning)
    evaluation = rng.sample([q for q in ids if q not in selected], n_evaluation)
    return dict(seed=seed, n_tuning=n_tuning, n_evaluation=n_evaluation,
                candidate_min=candidate_min, candidate_max=candidate_max,
                eligibility_rules=list(ELIGIBILITY_RULES),
                eligible_population_size=len(ids), tuning_query_ids=tuning,
                evaluation_query_ids=evaluation)


def distribution_summary(values: pd.Series) -> dict:
    """Descriptive counts only; linear-interpolated pandas quantiles."""
    if values.empty:
        return {}
    result = dict(min=float(values.min()), mean=float(values.mean()),
                  median=float(values.median()), max=float(values.max()))
    result.update({f'p{p}': float(values.quantile(p / 100))
                   for p in (25, 75, 90, 95, 99)})
    return result


def experiment_query_diagnostics(
    population: pd.DataFrame, papers: pd.DataFrame, graph: nx.DiGraph,
    query_ids: list[str],
) -> dict:
    """Cost/shape diagnostics; distinct abstract TEXTS include queries and reuse.

    Also report unique paper IDs to distinguish shared papers from identical text
    attached to different papers. No similarities, ranks or relevance metrics.
    """
    if len(set(query_ids)) != len(query_ids):
        raise ValueError('Duplicate query IDs')
    rows = population.set_index('query_id').loc[query_ids]
    if not rows.eligible.all():
        raise ValueError('Diagnostics require eligible queries')
    abstracts = _abstract_lookup(papers)
    needed = set(query_ids)
    for q in query_ids:
        needed.update(get_reference_list(graph, q) & abstracts.keys())
    needed = {p for p in needed if abstracts[p] is not None}
    texts = {abstracts[p] for p in needed}
    return dict(
        n_queries=len(rows), candidate_count=distribution_summary(rows.n_eligible_candidates),
        ground_truth=distribution_summary(rows.n_ground_truth),
        missing_abstract_fraction=distribution_summary(rows.missing_abstract_fraction),
        sum_eligible_candidates=int(rows.n_eligible_candidates.sum()),
        mean_eligible_candidates=float(rows.n_eligible_candidates.mean()),
        median_eligible_candidates=float(rows.n_eligible_candidates.median()),
        max_eligible_candidates=int(rows.n_eligible_candidates.max()),
        n_candidates_with_abstract=int(rows.n_candidates_with_abstract.sum()),
        n_candidates_without_abstract=int(rows.n_candidates_without_abstract.sum()),
        n_unique_abstracts_needed=len(texts), n_unique_papers_with_abstract=len(needed),
        unique_abstract_characters=sum(map(len, texts)),
        max_abstract_utf8_bytes=max((len(t.encode('utf-8')) for t in texts), default=0),
        embedding_float32_bytes=len(texts) * 1536 * 4,
    )


def validate_fixed_textual_support(papers: pd.DataFrame, word_vectors) -> dict:
    """Preflight existing Word2Vec means and TF-IDF-compatible tokens, no fit/API.

    Require valid titles globally: malformed candidate titles are data errors,
    not a reason to silently change candidate eligibility. Provider availability
    and future embedding responses cannot be guaranteed by an offline preflight.
    """
    from literature_recommender.word2vec import prepare_word2vec_corpus, document_vector, tokenize_abstract
    corpus = prepare_word2vec_corpus(papers)
    for p, title in papers[['paper_id', 'title']].itertuples(index=False, name=None):
        if not isinstance(title, str) or not tokenize_abstract(title):
            raise ValueError(f'Invalid title for {p!r}')
    valid, failures = set(), {}
    for p, text in corpus.texts.items():
        try:
            document_vector(tokenize_abstract(text), word_vectors)
        except ValueError as error:
            failures[p] = str(error)
        else:
            valid.add(p)
    return dict(valid_ids=valid, failures=failures,
                n_valid_abstracts=len(corpus.texts), n_missing_abstracts=len(corpus.missing_ids))
