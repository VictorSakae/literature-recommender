"""Stage 7C: offline grid search over the 100 frozen tuning queries only."""

import json
from pathlib import Path

import pandas as pd

from literature_recommender.grid_search import (
    REPRESENTATIONS, TIE_ATOL, read_tuning_scores, run_grid_search,
)
from literature_recommender.materialization import COMMON, TEXTUAL, file_hash, verify_hashes

FROZEN_HASHES = {
    'experiment_scores_common.csv': '9874c39e8349cdcb0fb9be4a53473a02b4008e5be70bc1c191563ad7d7347a73',
    'experiment_scores_tfidf.csv': '045849b70b037ac52cd034ae459e26359d4a2e17081337b2561a83ab96ec9c60',
    'experiment_scores_word2vec.csv': '9fe378647d775ae9d1675f08c4e0b2622b4593fedfebd9c1b21ac4b69ba92efe',
    'experiment_scores_embedding.csv': '3e740bb9b5c69e6bc74cb41d779ff4e893be8054cf73847f1a526974f2d8f1b7',
    'experiment_scores_manifest.json': 'f7a80d1bcd5b3f8ecfa4a5b68654039e9b4d25b156a4b4fd74a5eb98a060974f',
    'experiment_protocol.json': '66f7bed922959b2d5d163df359eca43652317d596cffcfadf43ffcde1195b4d4',
}


def main():
    output = Path('data/processed')
    paths = [output / name for name in ('grid_search_results.csv',
             'grid_search_query_results.csv', 'best_weights.json', 'grid_search_manifest.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Stage 7C outputs already exist; refusing to overwrite')
    inputs = {str(output / name): digest for name, digest in FROZEN_HASHES.items()}
    verify_hashes(inputs)
    protocol = json.loads((output / 'experiment_protocol.json').read_text())
    evidence = json.loads((output / 'experiment_scores_manifest.json').read_text())
    verify_hashes(evidence['artifact_sha256'])
    tuning_path = output / 'tuning_queries.csv'
    inputs[str(tuning_path)] = protocol['artifact_sha256'][str(tuning_path)]
    verify_hashes(inputs)
    queries = pd.read_csv(tuning_path)
    if (len(queries) != 100 or queries.query_id.tolist() != protocol['tuning_query_ids']
            or queries.query_id.duplicated().any() or not queries.eligible.all()):
        raise ValueError('Expected exactly the frozen 100 tuning IDs')
    common = read_tuning_scores(output / 'experiment_scores_common.csv', COMMON)
    representations = {name: read_tuning_scores(output / f'experiment_scores_{name}.csv', TEXTUAL)
                       for name in REPRESENTATIONS}
    if len(common) != protocol['diagnostics']['tuning']['sum_eligible_candidates']:
        raise ValueError('Frozen tuning pair total differs')
    results, individual, best, regressions = run_grid_search(common, representations, queries)
    if len(results) != 198 or len(individual) != 19800:
        raise ValueError('Incomplete grid results')
    verify_hashes(inputs)
    results.to_csv(paths[0], index=False)
    individual.to_csv(paths[1], index=False)
    paths[2].write_text(json.dumps(best, indent=2, allow_nan=False) + '\n')
    manifest = dict(
        stage='7C', split='tuning', input_sha256=inputs,
        protocol_reference=str(output / 'experiment_protocol.json'),
        protocol_sha256=inputs[str(output / 'experiment_protocol.json')],
        n_tuning_queries=100, n_tuning_pairs=len(common), tuning_query_ids=queries.query_id.tolist(),
        grid_step=0.1, n_weight_combinations=66, representations=list(REPRESENTATIONS),
        weight_order=['w_abstract', 'w_title', 'w_citation'],
        selection_metric='mean_recall_at_20', aggregation='equal-weight query mean, math.fsum',
        tie_policy=dict(abs_tol=TIE_ATOL, rel_tol=0.0, reference='maximum mean recall'),
        canonical_selection_policy='lexicographically smallest (w_abstract, w_title, w_citation)',
        candidate_tie_policy='score descending, paper_id ascending',
        auxiliary_metrics='diagnostic only; population std ddof=0; no role in selection',
        n_aggregate_results=len(results), n_query_results=len(individual),
        regression_checks=dict(extremes_and_060202='passed', tuning_query_ids=regressions),
        evaluation_used=False, external_calls=0, representations_recomputed=False,
        code_sha256={str(p): file_hash(p) for p in [Path(__file__),
            Path('src/literature_recommender/grid_search.py')]},
        output_sha256={str(p): file_hash(p) for p in paths[:3]},
    )
    paths[3].write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    verify_hashes(inputs)
    print(json.dumps(best, indent=2))
    print(json.dumps({str(p): file_hash(p) for p in paths}, indent=2))


if __name__ == '__main__':
    main()
