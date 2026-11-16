"""Paired analysis of persisted query metrics; no scoring/ranking/model access."""

from itertools import combinations

import numpy as np
import pandas as pd

REPRESENTATIONS = ('embedding', 'tfidf', 'word2vec')
CONDITIONS = ('controlled', 'optimized')
METRICS = ('recall_at_20', 'hit_at_1', 'hit_at_5', 'hit_at_10')
FEATURES = ('missing_abstract_fraction', 'n_candidates', 'n_ground_truth')


def validate_results(results, query_ids, weights):
    """Require all six paired groups, consistent pre-experiment data and weights."""
    expected = set(query_ids)
    if not expected or len(expected) != len(query_ids):
        raise ValueError('Expected query IDs must be unique and nonempty')
    if (set(results.condition) != set(CONDITIONS)
            or set(results.representation) != set(REPRESENTATIONS)
            or results.duplicated(['condition', 'representation', 'query_id']).any()):
        raise ValueError('Unexpected or duplicate condition/representation/query')
    numeric = [*FEATURES, *METRICS, 'w_abstract', 'w_title', 'w_citation']
    if not all(pd.api.types.is_numeric_dtype(results[c]) for c in numeric):
        raise ValueError('Analysis inputs must be numeric')
    if not np.isfinite(results[numeric].to_numpy()).all():
        raise ValueError('Analysis inputs must be finite')
    if not results[[*METRICS, 'missing_abstract_fraction']].apply(
            lambda s: s.between(0, 1).all()).all():
        raise ValueError('Metric or missing fraction out of bounds')
    if ((results.n_candidates < 20).any() or (results.n_ground_truth < 1).any()
            or (results.n_ground_truth > results.n_candidates).any()
            or not results.n_candidates.mod(1).eq(0).all()
            or not results.n_ground_truth.mod(1).eq(0).all()):
        raise ValueError('Invalid frozen pool or ground-truth size')
    reference = None
    for condition in CONDITIONS:
        for name in REPRESENTATIONS:
            group = results.loc[results.condition.eq(condition) & results.representation.eq(name)]
            if len(group) != len(expected) or set(group.query_id) != expected:
                raise ValueError('Unpaired or missing query IDs')
            if not group[['w_abstract', 'w_title', 'w_citation']].eq(
                    list(weights[condition][name])).all().all():
                raise ValueError('Frozen condition weights differ')
            features = group.set_index('query_id')[list(FEATURES)].sort_index()
            if reference is not None:
                pd.testing.assert_frame_equal(features, reference, check_exact=True)
            reference = features


def recompute_summary(results, frozen_summary):
    """Independent NumPy aggregation; fail before downstream comparisons on mismatch."""
    rows = []
    for (condition, name), group in results.groupby(['condition', 'representation'], sort=True):
        row = dict(condition=condition, representation=name, n_queries=len(group))
        row.update(group[['w_abstract', 'w_title', 'w_citation']].iloc[0].to_dict())
        for metric in METRICS:
            values = group[metric].to_numpy()
            for stat, value in [('mean', np.mean(values)), ('std', np.std(values, ddof=0)),
                                ('median', np.median(values)), ('min', np.min(values)),
                                ('max', np.max(values))]:
                row[f'{stat}_{metric}'] = float(value)
        rows.append(row)
    actual = pd.DataFrame(rows).set_index(['condition', 'representation']).sort_index()
    if frozen_summary.duplicated(['condition', 'representation']).any():
        raise ValueError('Duplicate frozen aggregate')
    expected = frozen_summary.set_index(['condition', 'representation']).sort_index()
    if not actual.index.equals(expected.index) or set(actual.columns) != set(expected.columns):
        raise ValueError('Frozen summary groups/schema differ')
    if not np.allclose(actual.to_numpy(), expected[actual.columns].to_numpy(), atol=1e-14, rtol=0):
        raise ValueError('Independent aggregates differ from frozen 7D summary')
    return actual.reset_index()


def aligned_values(left, right):
    """Align unique query-indexed finite numeric series, rejecting silent losses."""
    if (left.empty or left.index.has_duplicates or right.index.has_duplicates
            or set(left.index) != set(right.index)):
        raise ValueError('Paired series must have exactly the same unique query IDs')
    left, right = left.sort_index(), right.sort_index()
    if not np.isfinite(left.to_numpy()).all() or not np.isfinite(right.to_numpy()).all():
        raise ValueError('Paired values must be finite')
    return left, right


def paired_difference(left, right):
    left, right = aligned_values(left, right)
    return left - right


def difference_summary(delta):
    """Exact unrounded win/tie/loss; population std and linear quantiles."""
    values = delta.to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Differences must be nonempty and finite')
    wins, ties, losses = (int(np.sum(values > 0)), int(np.sum(values == 0)),
                          int(np.sum(values < 0)))
    return dict(n_queries=len(values), wins=wins, ties=ties, losses=losses,
        wins_pct=100 * wins / len(values), ties_pct=100 * ties / len(values),
        losses_pct=100 * losses / len(values), mean_difference=float(np.mean(values)),
        median_difference=float(np.median(values)), std_difference=float(np.std(values, ddof=0)),
        min_difference=float(np.min(values)), max_difference=float(np.max(values)),
        p25=float(np.quantile(values, .25)), p75=float(np.quantile(values, .75)),
        mean_absolute_difference=float(np.mean(np.abs(values))),
        median_absolute_difference=float(np.median(np.abs(values))))


def paired_comparisons(results):
    rows, differences = [], []
    for condition in CONDITIONS:
        frame = results.loc[results.condition.eq(condition)]
        for left, right in combinations(REPRESENTATIONS, 2):
            for metric in METRICS:
                delta = paired_difference(
                    frame.loc[frame.representation.eq(left)].set_index('query_id')[metric],
                    frame.loc[frame.representation.eq(right)].set_index('query_id')[metric])
                identity = dict(comparison_type='representations', condition=condition,
                                comparison=f'{left} - {right}', metric=metric)
                rows.append(dict(**identity, **difference_summary(delta)))
                differences.extend(dict(**identity, query_id=q, difference=float(value))
                                   for q, value in delta.items())
    return pd.DataFrame(rows), pd.DataFrame(differences)


def tuning_effect(results):
    rows, differences = [], []
    for name in REPRESENTATIONS:
        frame = results.loc[results.representation.eq(name)]
        for metric in METRICS:
            optimized = frame.loc[frame.condition.eq('optimized')].set_index('query_id')[metric]
            controlled = frame.loc[frame.condition.eq('controlled')].set_index('query_id')[metric]
            delta = paired_difference(optimized, controlled)
            identity = dict(comparison_type='tuning', condition='optimized - controlled',
                            comparison=name, metric=metric)
            stats = difference_summary(delta)
            stats['improved'] = stats.pop('wins')
            stats['worsened'] = stats.pop('losses')
            stats['improved_pct'] = stats.pop('wins_pct')
            stats['worsened_pct'] = stats.pop('losses_pct')
            rows.append(dict(**identity, mean_controlled=float(controlled.mean()),
                             mean_optimized=float(optimized.mean()), **stats))
            differences.extend(dict(**identity, query_id=q, difference=float(value))
                               for q, value in delta.items())
    return pd.DataFrame(rows), pd.DataFrame(differences)


def paired_bootstrap_ci(delta, *, seed=42, n_resamples=10000):
    """Marginal percentile 95% CI for a mean difference, resampling queries.

    Sort query IDs first. Reset the RNG per comparison so the same IDs are drawn
    across all comparisons. Intervals are not simultaneous or multiple-adjusted.
    """
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError('n_resamples must be a positive integer')
    if delta.index.has_duplicates or delta.empty:
        raise ValueError('Bootstrap requires unique query IDs')
    values = delta.sort_index().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Bootstrap values must be finite')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975], method='linear')
    return dict(n_queries=len(values), seed=seed, n_resamples=n_resamples,
                confidence_level=.95, method='paired percentile, linear quantiles',
                mean_difference=float(values.mean()), ci_low=float(low), ci_high=float(high))


def correlation(left, right):
    """Pearson and Spearman after strict query alignment; average ranks for ties."""
    left, right = aligned_values(left, right)
    if left.nunique() < 2 or right.nunique() < 2:
        return dict(n_queries=len(left), pearson=None, spearman=None, status='constant input')
    return dict(n_queries=len(left), pearson=float(np.corrcoef(left, right)[0, 1]),
        spearman=float(np.corrcoef(left.rank(method='average'), right.rank(method='average'))[0, 1]),
        status='ok')


def correlations(results, paired_differences):
    features = results.loc[results.condition.eq('controlled') &
                           results.representation.eq('embedding')].set_index('query_id')
    rows = []
    for condition in CONDITIONS:
        for name in REPRESENTATIONS:
            values = results.loc[results.condition.eq(condition) &
                                  results.representation.eq(name)].set_index('query_id').recall_at_20
            for feature in FEATURES:
                rows.append(dict(condition=condition, outcome=name, outcome_type='recall_at_20',
                                 feature=feature, **correlation(features[feature], values)))
    for (condition, comparison), group in paired_differences.loc[
            paired_differences.metric.eq('recall_at_20')].groupby(['condition', 'comparison'], sort=True):
        delta = group.set_index('query_id').difference
        for kind, values in [('signed_difference', delta), ('absolute_difference', delta.abs())]:
            rows.append(dict(condition=condition, outcome=comparison, outcome_type=kind,
                feature='missing_abstract_fraction', **correlation(features.missing_abstract_fraction, values)))
    return pd.DataFrame(rows)


def quartile_labels(feature):
    """Shared feature-only linear quartiles; keep equal values in the same bin.

    Q1 includes the minimum. Intervals are right-closed; repeated boundaries
    may leave empty groups, which must be reported rather than splitting ties.
    """
    if feature.empty or not np.isfinite(feature.to_numpy()).all():
        raise ValueError('Quartile features must be finite and nonempty')
    cuts = np.quantile(feature, [.25, .5, .75], method='linear')
    labels = pd.Series(np.searchsorted(cuts, feature.to_numpy(), side='left') + 1,
                       index=feature.index, name='quartile')
    return labels, [float(feature.min()), *map(float, cuts), float(feature.max())]


def quartile_summaries(results, paired_differences):
    features = results.loc[results.condition.eq('controlled') &
                           results.representation.eq('embedding')].set_index('query_id')
    rows = []
    for feature in FEATURES:
        labels, bounds = quartile_labels(features[feature])
        for condition in CONDITIONS:
            outcomes = []
            for name in REPRESENTATIONS:
                values = results.loc[results.condition.eq(condition) &
                                      results.representation.eq(name)].set_index('query_id').recall_at_20
                outcomes.append((name, 'recall_at_20', values))
            if feature == 'missing_abstract_fraction':
                selected = paired_differences.loc[paired_differences.condition.eq(condition) &
                                                  paired_differences.metric.eq('recall_at_20')]
                for comparison, group in selected.groupby('comparison', sort=True):
                    outcomes.append((comparison, 'paired_difference', group.set_index('query_id').difference))
            for outcome, kind, values in outcomes:
                values, aligned_labels = aligned_values(values, labels)
                for quartile in range(1, 5):
                    group = values.loc[aligned_labels.eq(quartile)]
                    rows.append(dict(feature=feature, condition=condition, outcome=outcome,
                        outcome_type=kind, quartile=f'Q{quartile}', lower=bounds[quartile - 1],
                        upper=bounds[quartile], n_queries=len(group),
                        mean=float(group.mean()) if len(group) else None,
                        mean_absolute=float(group.abs().mean()) if len(group) else None))
    return pd.DataFrame(rows)
