"""Stage 8A: analyze frozen 7D query metrics without scoring or ranking."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from literature_recommender.analysis import (
    CONDITIONS, REPRESENTATIONS, paired_comparisons, tuning_effect,
    paired_bootstrap_ci, correlations, quartile_summaries,
    validate_results, recompute_summary,
)
from literature_recommender.materialization import file_hash, verify_hashes

FROZEN_HASHES = {
    'evaluation_query_results.csv': '474afd8010e83d3dc0c45e1d4322dabf3a91267b81f496e4a4a1e4014040b9fc',
    'evaluation_top20.csv': '034fd004d33a9cdb3f660098abc64d12f5456322514794ea89dd8f4540111358',
    'evaluation_summary.csv': 'bf19a5073697849f6774dff54925b59a4762d4221a6cfdff33bdb61f9c392174',
    'evaluation_results.json': '4fa7c25fee24874cbbf983ea7f520e30960f9cbf96b768cd4a23606db84afed8',
    'evaluation_manifest.json': '60a4fcf4018cdfe7623efc60a11e21256fdd218eba0af4d2d8a324bbc8631583',
}
CSV_NAMES = ('analysis_aggregates', 'analysis_paired_comparisons', 'analysis_tuning_effect',
             'analysis_query_differences', 'analysis_correlations', 'analysis_quartiles',
             'analysis_bootstrap_ci')
FIGURE_NAMES = ('recall_summary', 'recall_distribution_controlled',
                'tuning_delta_distribution', 'recall_vs_missing_abstracts')
LABELS = dict(embedding='Embedding', tfidf='TF-IDF', word2vec='Word2Vec')
COLORS = dict(embedding='#245b96', tfidf='#b66a12', word2vec='#528252')


def records(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict('records')


def make_figures(results, summary, tuning_differences, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    paths = []

    def save(fig, name):
        path = directory / f'{name}.png'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    for shift, condition, color in [(-.18, 'controlled', '#3c6685'), (.18, 'optimized', '#b07832')]:
        values = [float(summary.loc[summary.condition.eq(condition) &
                       summary.representation.eq(n), 'mean_recall_at_20'].iloc[0]) for n in REPRESENTATIONS]
        bars = ax.bar(x + shift, values, width=.34, label=condition.upper(), color=color)
        ax.bar_label(bars, fmt='%.4f', padding=3)
    ax.set(xticks=x, xticklabels=[LABELS[n] for n in REPRESENTATIONS], ylim=(0, 1),
           ylabel='Mean recall@20', title='Evaluation: mesmas 100 queries')
    ax.legend(loc='upper right')
    fig.tight_layout()
    save(fig, 'recall_summary')

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [results.loc[results.condition.eq('controlled') & results.representation.eq(n),
                        'recall_at_20'].to_numpy() for n in REPRESENTATIONS]
    boxes = ax.boxplot(data, tick_labels=[LABELS[n] for n in REPRESENTATIONS], patch_artist=True)
    for box, n in zip(boxes['boxes'], REPRESENTATIONS):
        box.set_facecolor(COLORS[n]); box.set_alpha(.4)
    ax.set(ylabel='Recall@20 por query', ylim=(-.04, 1.04),
           title='CONTROLLED: distribuição nas 100 queries')
    fig.tight_layout()
    save(fig, 'recall_distribution_controlled')

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [tuning_differences.loc[tuning_differences.comparison.eq(n) &
            tuning_differences.metric.eq('recall_at_20'), 'difference'].to_numpy() for n in REPRESENTATIONS]
    ax.boxplot(data, tick_labels=[LABELS[n] for n in REPRESENTATIONS], showfliers=False)
    rng = np.random.default_rng(42)
    for i, (values, n) in enumerate(zip(data, REPRESENTATIONS), start=1):
        ax.scatter(i + rng.uniform(-.12, .12, len(values)), values, alpha=.5,
                   s=15, color=COLORS[n], edgecolors='none')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set(ylabel='Recall@20 OPTIMIZED − CONTROLLED',
           title='Efeito do tuning: todos os 100 deltas por representação')
    fig.tight_layout()
    save(fig, 'tuning_delta_distribution')

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for row, condition in enumerate(CONDITIONS):
        for col, n in enumerate(REPRESENTATIONS):
            group = results.loc[results.condition.eq(condition) & results.representation.eq(n)]
            ax = axes[row, col]
            ax.scatter(group.missing_abstract_fraction, group.recall_at_20, s=18,
                       alpha=.55, color=COLORS[n], edgecolors='none')
            ax.set(title=f'{condition.upper()} · {LABELS[n]}', xlim=(-.03, 1.03), ylim=(-.03, 1.03))
            if col == 0: ax.set_ylabel('Recall@20')
            if row == 1: ax.set_xlabel('Fração de abstracts ausentes')
    fig.suptitle('Associação descritiva; sem interpretação causal', fontsize=12)
    fig.tight_layout()
    save(fig, 'recall_vs_missing_abstracts')
    return paths


def interpretation(paired, tuning, corr):
    recall = paired.loc[paired.metric.eq('recall_at_20')]
    effects = tuning.loc[tuning.metric.eq('recall_at_20')].set_index('comparison')

    def comparison_text(comparison):
        texts = []
        for condition in CONDITIONS:
            r = recall.loc[recall.condition.eq(condition) & recall.comparison.eq(comparison)].iloc[0]
            texts.append(f'{condition}: média Δ={r.mean_difference:.6f}, mediana={r.median_difference:.6f}, '
                f'vitórias/empates/derrotas={r.wins}/{r.ties}/{r.losses}; '
                f'média |Δ|={r.mean_absolute_difference:.6f}')
        return '; '.join(texts)

    def effect_text(name):
        r = effects.loc[name]
        return (f'{LABELS[name]}: média Δ={r.mean_difference:.6f}, mediana={r.median_difference:.6f}; '
            f'melhoraram/empataram/pioraram={r.improved}/{r.ties}/{r.worsened}; '
            f'p25={r.p25:.6f}, p75={r.p75:.6f}, min={r.min_difference:.6f}, max={r.max_difference:.6f}')

    missing = corr.loc[corr.outcome_type.eq('absolute_difference')]
    missing_text = '; '.join(f'{r.condition}, {r.outcome}: Pearson={r.pearson:.3f}, '
                             f'Spearman={r.spearman:.3f}' for r in missing.itertuples())
    feature_text = []
    for feature in ('missing_abstract_fraction', 'n_candidates', 'n_ground_truth'):
        group = corr.loc[corr.feature.eq(feature) & corr.outcome_type.eq('recall_at_20')]
        feature_text.append(f'{feature}: Pearson de {group.pearson.min():.3f} a {group.pearson.max():.3f}; '
                            f'Spearman de {group.spearman.min():.3f} a {group.spearman.max():.3f}')
    return dict(
        Q1=comparison_text('embedding - tfidf'),
        Q2=comparison_text('embedding - word2vec'),
        Q3=effect_text('embedding') + '; ' + effect_text('tfidf'),
        Q4=effect_text('word2vec') + '; as contagens distinguem extensão da piora e sua magnitude; sem retuning.',
        Q5=missing_text + '; correlações de |Δ| descrevem convergência/divergência, sem causalidade.',
        Q6='; '.join(feature_text) + '; associações não ajustadas, sem interpretação causal.')


def main():
    output = Path('data/processed')
    figures = output / 'figures'
    protected = {str(output / n): h for n, h in FROZEN_HASHES.items()}
    verify_hashes(protected)
    manifest = json.loads((output / 'evaluation_manifest.json').read_text())
    protected.update(manifest['input_sha256'])
    verify_hashes(protected)
    verify_hashes(manifest['code_sha256'])
    best = json.loads((output / 'best_weights.json').read_text())
    weights = dict(controlled={n: manifest['controlled_weights'] for n in REPRESENTATIONS},
                   optimized={n: best[n]['canonical_best_weights'] for n in REPRESENTATIONS})
    csv_paths = [output / f'{n}.csv' for n in CSV_NAMES]
    json_path, manifest_path = output / 'analysis_summary.json', output / 'analysis_manifest.json'
    figure_paths = [figures / f'{n}.png' for n in FIGURE_NAMES]
    if any(p.exists() for p in [*csv_paths, json_path, manifest_path, *figure_paths]):
        raise FileExistsError('Stage 8A outputs already exist; refusing to overwrite')
    results = pd.read_csv(output / 'evaluation_query_results.csv', float_precision='round_trip')
    validate_results(results, manifest['evaluation_query_ids'], weights)
    if len(results) != 600 or len(manifest['evaluation_query_ids']) != 100:
        raise ValueError('Expected 600 results for 100 evaluation queries')
    frozen = pd.read_csv(output / 'evaluation_summary.csv', float_precision='round_trip')
    summary = recompute_summary(results, frozen)  # must pass before any further analysis
    print('Frozen aggregates independently reproduced: PASS', flush=True)
    paired, pair_diffs = paired_comparisons(results)
    tuning, tuning_diffs = tuning_effect(results)
    differences = pd.concat([pair_diffs, tuning_diffs], ignore_index=True)
    corr = correlations(results, pair_diffs)
    quartiles = quartile_summaries(results, pair_diffs)
    bootstrap = []
    for keys, group in differences.loc[differences.metric.eq('recall_at_20')].groupby(
            ['comparison_type', 'condition', 'comparison'], sort=True):
        bootstrap.append(dict(zip(['comparison_type', 'condition', 'comparison'], keys),
                              **paired_bootstrap_ci(group.set_index('query_id').difference)))
    bootstrap = pd.DataFrame(bootstrap)
    analysis = dict(n_queries=100, n_observations=600,
        conditions={c: int(results.condition.eq(c).sum()) for c in CONDITIONS},
        primary_metric='recall_at_20', complementary_metrics=['hit_at_1', 'hit_at_5', 'hit_at_10'],
        aggregation='population std ddof=0; linear quantiles; equal query weights; zeros retained',
        summary_validation='independent NumPy summaries reproduce frozen 7D within atol=1e-14, rtol=0',
        ties='exact equality of unrounded persisted values, no tolerance',
        main_table=records(summary),
        paired_recall=records(paired.loc[paired.metric.eq('recall_at_20')]),
        tuning_recall=records(tuning.loc[tuning.metric.eq('recall_at_20')]),
        correlations=records(corr), quartiles=records(quartiles),
        bootstrap=records(bootstrap), questions=interpretation(paired, tuning, corr),
        inference=dict(hypothesis_tests=False, p_values=False,
            bootstrap='95% marginal percentile CI; 10000 query resamples, seed=42, sorted IDs; same resamples across comparisons',
            multiplicity='Intervals are marginal, not simultaneous or multiple-comparison adjusted; no hypothesis rejection rule.',
            limitation='Exploratory query-resampling uncertainty; shared citation graph/candidates may induce dependence. No equivalence or causal claim.'),
        quartile_policy='linear feature-only quartiles on same 100 IDs; right-closed intervals, Q1 includes min; equal values never split; empty groups retained',
        recommendations=['Present controlled comparison first; optimized answers a different question.',
            'Present mean differences together with exact paired wins/ties/losses and absolute differences.',
            'Show Word2Vec tuning deltas without changing weights.',
            'Treat correlations and quartiles as descriptive associations, not causal explanations.',
            'Report uncertainty and the 100-query, dataset, graph, imputation and representation limitations.',
            'No direct numerical comparison with the paper or claim of universal superiority.'])
    verify_hashes(protected)
    for frame, path in zip((summary, paired, tuning, differences, corr, quartiles, bootstrap), csv_paths):
        frame.to_csv(path, index=False)
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    figures.mkdir(exist_ok=True)
    paths = [*csv_paths, json_path, *make_figures(results, summary, tuning_diffs, figures)]
    stage_manifest = dict(stage='8A', input_sha256=protected, n_queries=100, n_observations=600,
        frozen_7d_manifest_reference=str(output / 'evaluation_manifest.json'),
        policies={k: analysis[k] for k in ('aggregation', 'ties', 'inference', 'quartile_policy')},
        rankings_regenerated=False, weights_changed=False, representations_recomputed=False,
        external_calls=0, code_sha256={str(p): file_hash(p) for p in [Path(__file__),
            Path('src/literature_recommender/analysis.py')]},
        output_sha256={str(p): file_hash(p) for p in paths})
    manifest_path.write_text(json.dumps(stage_manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    verify_hashes(protected)
    print(json.dumps(analysis['questions'], indent=2, ensure_ascii=False))
    print(json.dumps({str(p): file_hash(p) for p in [*paths, manifest_path]}, indent=2))


if __name__ == '__main__':
    main()
