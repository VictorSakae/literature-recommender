"""Materialize stage 7B from frozen 7A. Run from repository root through uv."""
import argparse
import json
from pathlib import Path

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from openai import OpenAI

from literature_recommender.data import load_papers, load_citations
from literature_recommender.graph import build_citation_graph
from literature_recommender.materialization import (
    MODEL, DIMENSIONS, file_hash, verify_hashes, frozen_plans, common_scores,
    textual_scores, validate_tables, needed_texts, embedding_inventory,
    ensure_embeddings, persist_scores,
)
from literature_recommender.similarity import abstract_similarities, common_reference_count
from literature_recommender.tfidf import fit_tfidf, tfidf_abstract_similarities
from literature_recommender.word2vec import prepare_word2vec_corpus, word2vec_abstract_similarities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-only', action='store_true', help='Stop after local validation and embedding inventory')
    parser.add_argument('--embeddings-only', action='store_true', help='Resume using the three completed local score tables')
    args = parser.parse_args()
    if args.local_only and args.embeddings_only:
        parser.error('--local-only and --embeddings-only are mutually exclusive')
    output = Path('data/processed')
    protocol_path = output / 'experiment_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protected = {str(protocol_path): file_hash(protocol_path),
                 **protocol['source_sha256'], **protocol['artifact_sha256']}
    verify_hashes(protected)
    model_path = output / 'models/word2vec_cbow_100_seed42.model'
    model = Word2Vec.load(str(model_path))
    config = dict(sg=0, vector_size=100, window=5, min_count=5, epochs=5,
                  negative=5, hs=0, sample=0.001, seed=42)
    if any(getattr(model, k) != v for k, v in config.items()):
        raise ValueError('Frozen Word2Vec configuration mismatch')
    raw = Path('data/raw/Academic-Literature-Recommendation-in-Large-scale-Citation-Networks-main')
    papers = load_papers(raw / 'article_info.xlsx')
    graph = build_citation_graph(load_citations(raw / 'citation_network.csv'))
    population = pd.read_csv(output / 'experiment_population.csv')
    splits = {s: pd.read_csv(output / f'{s}_queries.csv') for s in ('tuning', 'evaluation')}
    plans = frozen_plans(papers, graph, protocol, population, splits)
    print('Protocol validated: 100 tuning, 100 evaluation, overlap 0', flush=True)
    if args.embeddings_only:
        common = pd.read_csv(output / 'experiment_scores_common.csv', float_precision='round_trip')
        frames = {name: pd.read_csv(output / f'experiment_scores_{name}.csv', float_precision='round_trip')
                  for name in ('tfidf', 'word2vec')}
        protected.update({str(output / f'experiment_scores_{name}.csv'):
                          file_hash(output / f'experiment_scores_{name}.csv')
                          for name in ('common', 'tfidf', 'word2vec')})
        for split, plan in plans:
            group = common.loc[common.query_id.eq(plan['query_id'])]
            if set(group.candidate_id) != plan['eligible_ids'] or not group.split.eq(split).all():
                raise ValueError('Completed common pairs differ from frozen pool')
            for row in group.itertuples(index=False):
                if row.relevant != (row.candidate_id in plan['relevant_ids']):
                    raise ValueError('Completed common ground truth differs')
                if row.candidate_has_abstract != (plan['abstracts'][row.candidate_id] is not None):
                    raise ValueError('Completed common abstract availability differs')
        validate_tables(common, frames)
        print('Completed local tables loaded and validated without regeneration', flush=True)
    else:
        common = common_scores(papers, graph, plans)
    for q, group in common.groupby('query_id', sort=False):
        raw_counts = np.array([common_reference_count(graph, q, p) for p in group.candidate_id])
        low, high = raw_counts.min(), raw_counts.max()
        expected = (raw_counts - low) / (high - low) if high != low else np.zeros(len(raw_counts))
        np.testing.assert_array_equal(group.citation_similarity.to_numpy(), expected)
    if not args.embeddings_only:
        validate_tables(common, {})
        persist_scores(common, output / 'experiment_scores_common.csv')
        print('Common scores persisted', flush=True)
        tfidf = fit_tfidf(papers)
        frames = {'tfidf': textual_scores(plans, lambda q, ids: tfidf_abstract_similarities(tfidf, q, ids))}
        validate_tables(common, frames)
        persist_scores(frames['tfidf'], output / 'experiment_scores_tfidf.csv')
        print('TF-IDF persisted', flush=True)
    corpus = prepare_word2vec_corpus(papers)
    if not args.embeddings_only:
        frames['word2vec'] = textual_scores(plans, lambda q, ids: word2vec_abstract_similarities(corpus, model.wv, q, ids))
        persist_scores(frames['word2vec'], output / 'experiment_scores_word2vec.csv')
    diagnostics = validate_tables(common, frames)
    print(json.dumps(dict(local_validation=diagnostics)), flush=True)
    # Regression uses similarities only. Historical ranks/metrics are never read.
    from literature_recommender.experiment import prepare_reference_query
    regression = [('independent', prepare_reference_query(papers, graph, 'paper_84898'))]
    checks = {} if args.embeddings_only else {'tfidf': lambda q, ids: tfidf_abstract_similarities(tfidf, q, ids),
              'word2vec': lambda q, ids: word2vec_abstract_similarities(corpus, model.wv, q, ids)}
    for name, scorer in checks.items():
        actual = textual_scores(regression, scorer).set_index('candidate_id')
        old = pd.read_csv(output / f'preliminary_{name}_paper_84898.csv',
                          usecols=['paper_id', 'abstract_similarity', 'abstract_imputed'],
                          float_precision='round_trip').set_index('paper_id').sort_index()
        actual = actual.sort_index()
        np.testing.assert_allclose(actual.abstract_similarity, old.abstract_similarity, atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(actual.abstract_imputed, old.abstract_imputed)
    texts = needed_texts(plans)
    if len(texts) != protocol['diagnostics']['combined']['n_unique_papers_with_abstract']:
        raise ValueError('Unique required papers differ from frozen diagnostic')
    cache = output / 'embeddings'
    inventory = embedding_inventory(texts, cache)
    print(json.dumps(dict(embedding_inventory=inventory)), flush=True)
    verify_hashes(protected)
    if args.local_only:
        return
    manifest_path = output / 'experiment_scores_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['protocol_sha256'] != protected[str(protocol_path)] or manifest['embedding_model'] != MODEL or manifest['embedding_dimensions'] != DIMENSIONS:
            raise ValueError('Existing manifest configuration differs')
        verify_hashes(manifest['artifact_sha256'])
        frames['embedding'] = pd.read_csv(output / 'experiment_scores_embedding.csv', float_precision='round_trip')
        validate_tables(common, frames)
        print('Complete materialization reused', flush=True)
        return
    if inventory['n_new_embeddings_required']:
        with OpenAI(timeout=90, max_retries=3) as client:
            vectors, generated = ensure_embeddings(texts, cache, client=client)
    else:
        vectors, generated = ensure_embeddings(texts, cache, client=None)
    frames['embedding'] = textual_scores(plans, lambda q, ids: abstract_similarities(
        vectors[q], {p: None if p in corpus.missing_ids else vectors[p] for p in ids}))
    diagnostics = validate_tables(common, frames)
    if diagnostics['n_queries'] != {'tuning': 100, 'evaluation': 100}:
        raise ValueError('Queries disappeared')
    persist_scores(frames['embedding'], output / 'experiment_scores_embedding.csv')
    verify_hashes(protected)
    names = ['common', 'tfidf', 'word2vec', 'embedding']
    hashes = {str(output / f'experiment_scores_{n}.csv'): file_hash(output / f'experiment_scores_{n}.csv') for n in names}
    if args.embeddings_only:
        params = json.loads((output / 'preliminary_tfidf_paper_84898_summary.json').read_text())['vectorizer_parameters']
    else:
        params = tfidf.vectorizer.get_params()
        params['dtype'] = np.dtype(params['dtype']).name
    manifest = dict(protocol_reference=str(protocol_path), protocol_sha256=protected[str(protocol_path)],
                    n_queries_tuning=100, n_queries_evaluation=100, overlap=0, **diagnostics,
                    n_pairs_by_artifact={n: len(common) for n in names},
                    relevant_count=int(common.relevant.sum()),
                    title_diagnostics=common.title_similarity.agg(['min', 'max', 'mean']).to_dict(),
                    citation_diagnostics=common.citation_similarity.agg(['min', 'max', 'mean']).to_dict(),
                    citation_validation='raw outgoing intersections and full-pool min-max independently checked for every query',
                    imputation_validation='independent math.fsum of real scores checked for every query in all representations',
                    tfidf_configuration_reference='src/literature_recommender/tfidf.py:fit_tfidf',
                    tfidf_configuration=params, word2vec_model_reference=str(model_path),
                    word2vec_model_sha256=file_hash(model_path), word2vec_configuration=config,
                    embedding_model=MODEL, embedding_dimensions=DIMENSIONS,
                    embedding_cache=str(cache), embedding_inventory_before=inventory,
                    embedding_generation=generated, artifact_sha256=hashes)
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
