"""Stage 7D: two predeclared conditions on frozen evaluation evidence only."""

import csv
from io import StringIO
import json
from math import fsum

import numpy as np
import pandas as pd

from literature_recommender.evaluation import evaluate_ranking
from literature_recommender.materialization import COMMON, KEYS, validate_tables
from literature_recommender.recommender import hybrid_scores, rank_scores

REPRESENTATIONS = ('embedding', 'tfidf', 'word2vec')
CONTROLLED = (0.6, 0.2, 0.2)
EXPECTED_OPTIMIZED = dict(embedding=(0.6, 0.1, 0.3), tfidf=(0.6, 0.1, 0.3),
                          word2vec=(0.3, 0.4, 0.3))
METRICS = {'hit@1': 'hit_at_1', 'hit@5': 'hit_at_5', 'hit@10': 'hit_at_10',
           'recall@20': 'recall_at_20'}
WEIGHT_COLUMNS = ['w_abstract', 'w_title', 'w_citation']


def load_best_weights(path):
    """Read canonical weights, validate the frozen selection; never select again."""
    artifact = json.loads(path.read_text())
    if set(artifact) != set(REPRESENTATIONS):
        raise ValueError('Unexpected optimized representations')
    for name, expected in EXPECTED_OPTIMIZED.items():
        weights = artifact[name]['canonical_best_weights']
        if (any(type(w) not in (int, float) for w in weights)
                or tuple(weights) != expected):
            raise ValueError(f'Frozen optimized weights differ: {name}')
        mean = artifact[name]['best_mean_recall_at_20']
        if type(mean) not in (int, float) or not np.isfinite(mean) or not 0 <= mean <= 1:
            raise ValueError('Invalid persisted tuning mean')
    return artifact


def read_evaluation_scores(path, columns):
    """Discard tuning rows before numeric conversion, retaining evaluation only."""
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    with open(path, newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != columns:
            raise ValueError(f'Unexpected score schema: {path}')
        for row in reader:
            if row['split'] not in ('tuning', 'evaluation'):
                raise ValueError(f'Unexpected split: {path}')
            if row['split'] == 'evaluation':
                writer.writerow([row[c] for c in columns])
    buffer.seek(0)
    return pd.read_csv(buffer, float_precision='round_trip',
                       dtype={c: str for c in KEYS}, keep_default_na=False)


def prepare_evaluation_tables(common, representations, queries):
    """Validate exact IDs, full pools/truth, missingness and strict 1:1 keys."""
    if set(representations) != set(REPRESENTATIONS):
        raise ValueError('Exactly three frozen representations required')
    if queries.empty or queries.query_id.duplicated().any():
        raise ValueError('Evaluation query IDs must be unique and nonempty')
    for frame in [common, *representations.values()]:
        if not frame.split.eq('evaluation').all():
            raise ValueError('Only evaluation rows may enter final evaluation')
        if set(frame.query_id) != set(queries.query_id):
            raise ValueError('Evaluation query IDs differ from frozen IDs')
    validate_tables(common, representations)
    metadata = queries.set_index('query_id')
    for q, group in common.groupby('query_id', sort=True):
        if len(group) < 20 or not group.relevant.any():
            raise ValueError('Protocol requires >=20 candidates and nonempty ground truth')
        if (len(group) != metadata.loc[q, 'n_eligible_candidates']
                or int(group.relevant.sum()) != metadata.loc[q, 'n_ground_truth']):
            raise ValueError('Frozen candidate or ground truth count differs')
        fraction = float((~group.candidate_has_abstract).sum()) / len(group)
        if not np.isclose(fraction, metadata.loc[q, 'missing_abstract_fraction'],
                          atol=1e-15, rtol=0):
            raise ValueError('Frozen missing abstract fraction differs')
    tables = {}
    for name in REPRESENTATIONS:
        joined = common.merge(representations[name], on=KEYS, how='outer',
                              validate='one_to_one', indicator=True)
        if not joined['_merge'].eq('both').all():
            raise ValueError(f'Unmatched pairs: {name}')
        joined = joined.drop(columns='_merge').sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(joined[COMMON],
                                      common.sort_values(KEYS).reset_index(drop=True))
        tables[name] = joined
    return tables


def summarize_query_results(query_results):
    """Equal-query aggregation, including every zero; population std ddof=0."""
    rows = []
    for (condition, name), group in query_results.groupby(['condition', 'representation'], sort=True):
        if group.query_id.duplicated().any():
            raise ValueError('Duplicate evaluation query result')
        weights = group[WEIGHT_COLUMNS].drop_duplicates()
        if len(weights) != 1:
            raise ValueError('Weights changed within a condition')
        row = dict(condition=condition, representation=name,
                   **weights.iloc[0].to_dict(), n_queries=len(group))
        for metric in METRICS.values():
            values = group[metric].tolist()
            row.update({f'mean_{metric}': fsum(values) / len(values),
                        f'std_{metric}': float(np.std(values, ddof=0)),
                        f'median_{metric}': float(np.median(values)),
                        f'min_{metric}': min(values), f'max_{metric}': max(values)})
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_comparisons(summary, best):
    """Subtract predeclared conditions and persisted tuning means, no selection."""
    results = {condition: {} for condition in ('controlled', 'optimized')}
    deltas, gaps = {}, {}
    for name in REPRESENTATIONS:
        rows = summary.loc[summary.representation.eq(name)].set_index('condition')
        for condition in results:
            results[condition][name] = rows.loc[condition].drop('representation').to_dict()
        deltas[name] = {f'delta_mean_{metric}': float(
            rows.loc['optimized', f'mean_{metric}'] - rows.loc['controlled', f'mean_{metric}'])
            for metric in METRICS.values()}
        tuning = best[name]['best_mean_recall_at_20']
        evaluation = float(rows.loc['optimized', 'mean_recall_at_20'])
        gaps[name] = dict(tuning_best_mean_recall_at_20=tuning,
                          optimized_evaluation_mean_recall_at_20=evaluation,
                          tuning_to_evaluation_difference=evaluation - tuning)
    return dict(**results, optimized_vs_controlled_deltas=deltas,
                tuning_to_evaluation_comparison=gaps,
                interpretation='Descriptive only; no significance test or condition/weight selection.')


def _regression_check(indexed, weights, scores, ranking, relevant, metrics):
    """Independent score, full-ranking and top-k metric regression, not a new API."""
    a, t, c = weights
    manual = {p: a * row.abstract_similarity + t * row.title_similarity
              + c * row.citation_similarity for p, row in indexed.iterrows()}
    ordered = sorted(manual.items(), key=lambda item: (-item[1], item[0]))
    if scores != manual or ranking != ordered:
        raise AssertionError('Manual weighted score or full ranking differs')
    ids = [p for p, _ in ordered]
    for k in (1, 5, 10, 20):
        hits = len(set(ids[:k]) & relevant)
        expected = hits / (len(relevant) if k == 20 else k)
        if metrics['recall@20' if k == 20 else f'hit@{k}'] != expected:
            raise AssertionError('Independent top-k metric differs')


def run_final_evaluation(common, representations, queries, best):
    """Evaluate exactly two fixed conditions, retaining metrics and top-20 only."""
    if (set(best) != set(REPRESENTATIONS) or any(
            tuple(best[n]['canonical_best_weights']) != EXPECTED_OPTIMIZED[n]
            for n in REPRESENTATIONS)):
        raise ValueError('Frozen optimized weights differ')
    tables = prepare_evaluation_tables(common, representations, queries)
    checked_ids = sorted(queries.query_id)[:3]
    query_rows, top_rows = [], []
    for condition in ('controlled', 'optimized'):
        for name in REPRESENTATIONS:
            weights = CONTROLLED if condition == 'controlled' else tuple(best[name]['canonical_best_weights'])
            for q, group in tables[name].groupby('query_id', sort=True):
                indexed = group.set_index('candidate_id')
                relevant = set(indexed.index[indexed.relevant])
                scores = hybrid_scores(indexed.abstract_similarity.to_dict(),
                                       indexed.title_similarity.to_dict(),
                                       indexed.citation_similarity.to_dict(), weights=weights)
                ranking = rank_scores(scores)  # rank full frozen pool before any top-k
                metrics = evaluate_ranking([p for p, _ in ranking], relevant)
                if q in checked_ids:
                    _regression_check(indexed, weights, scores, ranking, relevant, metrics)
                query_rows.append(dict(condition=condition, representation=name,
                    **dict(zip(WEIGHT_COLUMNS, weights)), query_id=q,
                    n_candidates=len(group), n_ground_truth=len(relevant),
                    missing_abstract_fraction=float((~group.candidate_has_abstract).sum()) / len(group),
                    **{METRICS[k]: value for k, value in metrics.items()}))
                for rank, (p, score) in enumerate(ranking[:20], start=1):
                    row = indexed.loc[p]
                    top_rows.append(dict(condition=condition, representation=name, query_id=q,
                        rank=rank, candidate_id=p, relevant=bool(row.relevant),
                        weighted_similarity=score, abstract_similarity=row.abstract_similarity,
                        title_similarity=row.title_similarity, citation_similarity=row.citation_similarity,
                        abstract_imputed=bool(row.abstract_imputed)))
    query_results, top20 = pd.DataFrame(query_rows), pd.DataFrame(top_rows)
    summary = summarize_query_results(query_results)
    return query_results, top20, summary, descriptive_comparisons(summary, best), checked_ids
