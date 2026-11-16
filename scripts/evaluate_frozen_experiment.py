"""Stage 7D: offline controlled/optimized evaluation with frozen tuning weights."""

import json
from pathlib import Path

import pandas as pd

from literature_recommender.final_evaluation import (
    CONTROLLED, REPRESENTATIONS, load_best_weights,
    read_evaluation_scores, run_final_evaluation,
)
from literature_recommender.materialization import COMMON, TEXTUAL, file_hash, verify_hashes

FROZEN_HASHES = {
    'experiment_scores_common.csv': '9874c39e8349cdcb0fb9be4a53473a02b4008e5be70bc1c191563ad7d7347a73',
    'experiment_scores_tfidf.csv': '045849b70b037ac52cd034ae459e26359d4a2e17081337b2561a83ab96ec9c60',
    'experiment_scores_word2vec.csv': '9fe378647d775ae9d1675f08c4e0b2622b4593fedfebd9c1b21ac4b69ba92efe',
    'experiment_scores_embedding.csv': '3e740bb9b5c69e6bc74cb41d779ff4e893be8054cf73847f1a526974f2d8f1b7',
    'experiment_scores_manifest.json': 'f7a80d1bcd5b3f8ecfa4a5b68654039e9b4d25b156a4b4fd74a5eb98a060974f',
    'experiment_protocol.json': '66f7bed922959b2d5d163df359eca43652317d596cffcfadf43ffcde1195b4d4',
    'grid_search_results.csv': '63a734e5d10884384c54ab42e8ebeed85d190fe06cb024638e266a556207019e',
    'grid_search_query_results.csv': '6b804b7de00c53f932c5a76a7c8b8c1b7f76f2b41a12cbffebcdd154f4d5ff18',
    'best_weights.json': '9070d112bc2bdef06e166171c3f9a88ba5af94920b45654a7f16094ed8ef3c7c',
    'grid_search_manifest.json': '8309eaa9548383e2c09caf0521008328b16ce2335ab40df98206a572fa05602d',
}


def main():
    output = Path('data/processed')
    paths = [output / name for name in ('evaluation_query_results.csv', 'evaluation_top20.csv',
             'evaluation_summary.csv', 'evaluation_results.json', 'evaluation_manifest.json')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Stage 7D outputs already exist; refusing to overwrite')
    inputs = {str(output / n): h for n, h in FROZEN_HASHES.items()}
    verify_hashes(inputs)
    protocol = json.loads((output / 'experiment_protocol.json').read_text())
    evidence = json.loads((output / 'experiment_scores_manifest.json').read_text())
    tuning_manifest = json.loads((output / 'grid_search_manifest.json').read_text())
    verify_hashes(evidence['artifact_sha256'])
    verify_hashes(tuning_manifest['input_sha256'])
    verify_hashes(tuning_manifest['output_sha256'])
    query_path = output / 'evaluation_queries.csv'
    inputs[str(query_path)] = protocol['artifact_sha256'][str(query_path)]
    verify_hashes(inputs)
    queries = pd.read_csv(query_path, float_precision='round_trip')
    if (len(queries) != 100 or queries.query_id.tolist() != protocol['evaluation_query_ids']
            or queries.query_id.duplicated().any() or not queries.eligible.all()
            or set(queries.query_id) & set(protocol['tuning_query_ids'])):
        raise ValueError('Expected exactly the frozen 100 disjoint evaluation IDs')
    best_path = output / 'best_weights.json'
    best = load_best_weights(best_path)
    common = read_evaluation_scores(output / 'experiment_scores_common.csv', COMMON)
    representations = {n: read_evaluation_scores(output / f'experiment_scores_{n}.csv', TEXTUAL)
                       for n in REPRESENTATIONS}
    if len(common) != 22873 or len(common) != protocol['diagnostics']['evaluation']['sum_eligible_candidates']:
        raise ValueError('Expected exactly 22873 frozen evaluation pairs')
    individual, top20, summary, results, checked_ids = run_final_evaluation(
        common, representations, queries, best)
    if len(individual) != 600 or len(top20) != 12000 or len(summary) != 6:
        raise ValueError('Incomplete evaluation output')
    verify_hashes(inputs)
    for frame, path in zip((individual, top20, summary), paths[:3]):
        frame.to_csv(path, index=False)
    paths[3].write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    manifest = dict(stage='7D', split='evaluation', input_sha256=inputs,
        protocol_reference=str(output / 'experiment_protocol.json'),
        protocol_sha256=inputs[str(output / 'experiment_protocol.json')],
        best_weights_reference=str(best_path), best_weights_sha256=inputs[str(best_path)],
        n_evaluation_queries=100, n_pairs=len(common), evaluation_query_ids=queries.query_id.tolist(),
        representations=list(REPRESENTATIONS), controlled_weights=list(CONTROLLED),
        optimized_weights={n: best[n]['canonical_best_weights'] for n in REPRESENTATIONS},
        weight_order=['w_abstract', 'w_title', 'w_citation'],
        metric_reference='src/literature_recommender/evaluation.py:evaluate_ranking',
        metric_definitions=dict(hit_at_k='|top-k intersect relevant| / k, k=1,5,10',
                                recall_at_20='|top-20 intersect relevant| / |relevant|'),
        aggregation='equal-query mean with math.fsum; population std ddof=0; zeros retained',
        ranking='full frozen pool; score descending then paper_id ascending',
        primary_endpoint='mean_recall_at_20',
        n_query_results=len(individual), n_top20_rows=len(top20), n_summary_rows=len(summary),
        regression_checks=dict(status='passed', evaluation_query_ids=checked_ids,
            checks='independent weighted scores/full ranking and top1/5/10/20 metrics; both conditions, all representations'),
        tuning_scores_used=False, tuning_metrics_recomputed=False, weights_selected_on_evaluation=False,
        representations_recomputed=False, external_calls=0,
        descriptive_comparisons_only=True,
        code_sha256={str(p): file_hash(p) for p in [Path(__file__),
            *[Path('src/literature_recommender') / n for n in
              ('final_evaluation.py', 'evaluation.py', 'recommender.py')]]},
        output_sha256={str(p): file_hash(p) for p in paths[:4]})
    paths[4].write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    verify_hashes(inputs)
    print(summary.to_string(index=False))
    print(json.dumps(results, indent=2))
    print(json.dumps({str(p): file_hash(p) for p in paths}, indent=2))


if __name__ == '__main__':
    main()
