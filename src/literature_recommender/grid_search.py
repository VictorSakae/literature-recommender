"""Stage 7C: tune frozen evidence only; never acquire representations."""

import csv
from io import StringIO
from math import fsum, isclose

import numpy as np
import pandas as pd

from literature_recommender.evaluation import recall_at_k
from literature_recommender.materialization import COMMON, KEYS, TEXTUAL, validate_tables
from literature_recommender.recommender import hybrid_scores, rank_scores

REPRESENTATIONS = ('embedding', 'tfidf', 'word2vec')
WEIGHT_COLUMNS = ['w_abstract', 'w_title', 'w_citation']
TIE_ATOL = 1e-12


def weight_grid():
    """66 unique simplex points generated with integer tenths."""
    return [(a / 10, t / 10, (10 - a - t) / 10)
            for a in range(11) for t in range(11 - a)]


def read_tuning_scores(path, columns):
    """Stream CSV rows; discard evaluation before numeric/bool conversion.

    Only tuning records enter the returned frame. Unknown splits fail rather
    than silently dropping a malformed tuning label. No input file is written.
    """
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
            if row['split'] == 'tuning':
                writer.writerow([row[c] for c in columns])
    buffer.seek(0)
    return pd.read_csv(buffer, float_precision='round_trip',
                       dtype={c: str for c in KEYS}, keep_default_na=False)


def prepare_tuning_tables(common, representations, queries):
    """Validate exact membership, full-pool counts and strict one-to-one joins."""
    if set(representations) != set(REPRESENTATIONS):
        raise ValueError('Exactly embedding, tfidf and word2vec are required')
    if queries.query_id.duplicated().any() or queries.empty:
        raise ValueError('Tuning query IDs must be unique and nonempty')
    expected_ids = set(queries.query_id)
    for frame in [common, *representations.values()]:
        if not frame.split.eq('tuning').all():
            raise ValueError('Only tuning rows may enter grid search')
        if set(frame.query_id) != expected_ids:
            raise ValueError('Tuning query IDs differ from frozen IDs')
    validate_tables(common, representations)
    metadata = queries.set_index('query_id')
    for q, group in common.groupby('query_id', sort=True):
        if len(group) < 20 or not group.relevant.any():
            raise ValueError('Protocol requires >=20 candidates and nonempty ground truth')
        if (len(group) != metadata.loc[q, 'n_eligible_candidates']
                or int(group.relevant.sum()) != metadata.loc[q, 'n_ground_truth']):
            raise ValueError('Frozen pool or ground truth count differs')
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


def _query_components(frame):
    for q, group in frame.groupby('query_id', sort=True):
        indexed = group.set_index('candidate_id')
        yield (q, indexed.abstract_similarity.to_dict(),
               indexed.title_similarity.to_dict(), indexed.citation_similarity.to_dict(),
               set(indexed.index[indexed.relevant]))


def regression_checks(tables, sample_size=3):
    """Independent full-ranking checks on first sorted tuning IDs only."""
    checked = {}
    for name, frame in tables.items():
        checked[name] = []
        for q, abstracts, titles, citations, _ in list(_query_components(frame))[:sample_size]:
            for weights, component in [((1., 0., 0.), abstracts),
                                       ((0., 1., 0.), titles), ((0., 0., 1.), citations)]:
                actual = hybrid_scores(abstracts, titles, citations, weights=weights)
                if actual != component:
                    raise AssertionError('Extreme weights changed component scores')
                expected = sorted(component.items(), key=lambda item: (-item[1], item[0]))
                if rank_scores(actual) != expected:
                    raise AssertionError('Extreme ranking differs')
            expected_scores = {p: .6 * abstracts[p] + .2 * titles[p] + .2 * citations[p]
                               for p in abstracts}
            actual = hybrid_scores(abstracts, titles, citations, weights=(.6, .2, .2))
            if actual != expected_scores or rank_scores(actual) != sorted(
                    expected_scores.items(), key=lambda item: (-item[1], item[0])):
                raise AssertionError('0.6/0.2/0.2 regression differs')
            checked[name].append(q)
    return checked


def select_best(results):
    """Maximize only mean recall; retain every near-maximum, lexicographic canonical."""
    best = {}
    for name in REPRESENTATIONS:
        subset = results.loc[results.representation.eq(name)]
        maximum = float(subset.mean_recall_at_20.max())
        configs = sorted(tuple(float(row[c]) for c in WEIGHT_COLUMNS)
                         for _, row in subset.iterrows()
                         if isclose(row.mean_recall_at_20, maximum,
                                    abs_tol=TIE_ATOL, rel_tol=0.0))
        best[name] = dict(best_mean_recall_at_20=maximum,
                          n_best_tied_configs=len(configs),
                          all_best_configs=[list(c) for c in configs],
                          canonical_best_weights=list(configs[0]))
    return best


def run_grid_search(common, representations, queries):
    """Rank every full pool via existing APIs; average recall equally per query.

    All auxiliary statistics are diagnostics only; std uses population ddof=0.
    Input frames are validated without modification. No hit metrics are called.
    """
    tables = prepare_tuning_tables(common, representations, queries)
    regressions = regression_checks(tables)
    aggregate, individual = [], []
    for name in REPRESENTATIONS:
        components = list(_query_components(tables[name]))
        for weights in weight_grid():
            recalls = []
            for q, abstracts, titles, citations, relevant in components:
                scores = hybrid_scores(abstracts, titles, citations, weights=weights)
                ranking = rank_scores(scores)  # full pool, top-20 only inside recall
                recall = recall_at_k([p for p, _ in ranking], relevant, 20)
                recalls.append(recall)
                individual.append((name, *weights, q, recall))
            aggregate.append((name, *weights, len(recalls), fsum(recalls) / len(recalls),
                              float(np.std(recalls, ddof=0)), float(np.median(recalls)),
                              min(recalls), max(recalls)))
        print(f'{name}: 66 configurations, {len(components)} tuning queries', flush=True)
    results = pd.DataFrame(aggregate, columns=['representation', *WEIGHT_COLUMNS,
        'n_queries', 'mean_recall_at_20', 'std_recall_at_20', 'median_recall_at_20',
        'min_recall_at_20', 'max_recall_at_20'])
    query_results = pd.DataFrame(individual, columns=['representation', *WEIGHT_COLUMNS,
                                                    'query_id', 'recall_at_20'])
    return results, query_results, select_best(results), regressions
