"""Stage 7A offline diagnostic, then explicit freeze; never computes similarities."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform

from gensim.models import Word2Vec
import pandas as pd

from literature_recommender.data import load_papers, load_citations
from literature_recommender.graph import build_citation_graph
from literature_recommender.experiment import (
    build_experiment_population, split_experiment_queries,
    experiment_query_diagnostics, distribution_summary, validate_fixed_textual_support,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-tuning', type=int, default=100)
    parser.add_argument('--n-evaluation', type=int, default=100)
    parser.add_argument('--candidate-max', type=int, default=None)
    parser.add_argument('--cost-justification', default=None)
    parser.add_argument('--freeze', action='store_true', help='Persist only after reviewing offline costs')
    args = parser.parse_args()
    if args.freeze and (args.candidate_max is not None or
                        (args.n_tuning, args.n_evaluation) != (100, 100)) and not args.cost_justification:
        parser.error('Changed bounds/sizes require --cost-justification before freezing')
    raw = Path('data/raw/Academic-Literature-Recommendation-in-Large-scale-Citation-Networks-main')
    processed = Path('data/processed')
    outputs = [processed / name for name in ('experiment_population.csv', 'tuning_queries.csv',
                                            'evaluation_queries.csv', 'experiment_protocol.json')]
    if args.freeze and any(p.exists() for p in outputs):
        raise FileExistsError('Frozen protocol exists; preserve it. Bug corrections require documentation.')
    model_path = processed / 'models/word2vec_cbow_100_seed42.model'
    sources = [raw / 'article_info.xlsx', raw / 'citation_network.csv', model_path,
               Path('uv.lock'), Path('docs/papers/2025_Liu_AcademicLiteratureRecommendation.pdf')]
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in sources}
    papers = load_papers(sources[0])
    graph = build_citation_graph(load_citations(sources[1]))
    print(json.dumps(dict(n_articles=len(papers), n_graph_nodes=len(graph))), flush=True)
    model = Word2Vec.load(str(model_path))
    support = validate_fixed_textual_support(papers, model.wv)
    print(json.dumps({k: v for k, v in support.items() if k != 'valid_ids'}), flush=True)
    population = build_experiment_population(papers, graph, textual_valid_ids=support['valid_ids'])
    eligible = population.loc[population.eligible]
    exclusions = population.loc[~population.eligible, 'exclusion_reason'].value_counts().to_dict()
    structural_reasons = {'', 'NO_VALID_TEXTUAL_COMPARISON', 'INVALID_CANDIDATE_REPRESENTATION'}
    summary = dict(n_articles=len(papers), n_graph_nodes=len(graph),
                   n_considered_queries=len(population),
                   n_queries_after_structural_criteria=int(population.exclusion_reason.isin(structural_reasons).sum()),
                   eligible_population_size=len(eligible), exclusion_counts=exclusions,
                   candidate_distribution_before_cap=distribution_summary(eligible.n_eligible_candidates),
                   ground_truth_distribution=distribution_summary(eligible.n_ground_truth),
                   missing_abstract_distribution=distribution_summary(eligible.missing_abstract_fraction))
    print(json.dumps(summary, indent=2), flush=True)
    split = split_experiment_queries(population, n_tuning=args.n_tuning, n_evaluation=args.n_evaluation,
                                     candidate_max=args.candidate_max)
    diagnostics = {name: experiment_query_diagnostics(population, papers, graph, split[f'{name}_query_ids'])
                   for name in ('tuning', 'evaluation')}
    diagnostics['combined'] = experiment_query_diagnostics(
        population, papers, graph, split['tuning_query_ids'] + split['evaluation_query_ids'])
    print(json.dumps(diagnostics, indent=2), flush=True)
    assert not set(split['tuning_query_ids']) & set(split['evaluation_query_ids'])
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    if not args.freeze:
        print('Diagnostic only. Review cost before --freeze.', flush=True)
        return
    protocol = dict(
        **split, exclusion_counts=exclusions, population_diagnostics=summary,
        diagnostics=diagnostics, source_sha256=hashes,
        status='frozen for 7B+', python_version=platform.python_version(),
        sampling='random.Random(seed).sample: sorted eligible IDs for tuning, sorted remaining IDs for evaluation',
        quantiles='pandas linear interpolation',
        ground_truth_rule='direct outgoing references intersect metadata IDs minus query',
        candidate_list_rule='full REFERENCE LIST, outgoing depth 1-3 minus query; metadata eligibility; no CITATION LIST',
        text_representation_policy=dict(embedding='fixed pretrained text-embedding-3-small, 1536 dimensions',
                                        tfidf='global valid abstract corpus fit; fixed vocabulary/IDF',
                                        word2vec='global valid abstract corpus training; existing fixed CBOW model',
                                        ground_truth_used_for_fit=False),
        textual_preflight=dict(n_valid_abstracts=support['n_valid_abstracts'],
                               failures=support['failures'], titles='all metadata titles have tokens',
                               embedding='nonblank text checked offline; provider/response failures remain experimental errors'),
        missing_policy='keep missing candidate abstracts; mean imputation within full eligible list later',
        cost_decision=args.cost_justification or '100+100, no maximum: sampled full pools and unique-text memory fit a modest offline experiment; no API calls now.',
        population_preserved_fraction=split['eligible_population_size'] / len(eligible),
        split_unit='query; candidate overlap allowed',
        weight_selection='tuning only; freeze weights before evaluation; same IDs/pools/truth for all three representations',
        freeze_policy='No changes after performance; only documented bug/eligibility/population/cost corrections before experiments',
        population_manifest='eligible rows only; exclusions aggregated over metadata UNION graph IDs; reproducible from sources',
    )
    eligible.to_csv(outputs[0], index=False, mode='x')
    indexed = population.set_index('query_id', drop=False)
    for name, path in zip(('tuning', 'evaluation'), outputs[1:3]):
        indexed.loc[split[f'{name}_query_ids']].to_csv(path, index=False, mode='x')
    protocol['artifact_sha256'] = {str(p): sha256(p.read_bytes()).hexdigest() for p in outputs[:3]}
    with outputs[3].open('x') as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
