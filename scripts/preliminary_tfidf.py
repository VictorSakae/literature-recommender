"""Stage 6A: fixed N=1 sanity check; run from repository root through uv."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from literature_recommender.data import load_papers, load_citations, validate_data
from literature_recommender.evaluation import evaluate_ranking
from literature_recommender.experiment import prepare_reference_query
from literature_recommender.graph import build_citation_graph
from literature_recommender.recommender import hybrid_scores, rank_scores
from literature_recommender.similarity import citation_similarities, title_similarity
from literature_recommender.tfidf import fit_tfidf, tfidf_abstract_similarities


def main():
    raw = Path('data/raw/Academic-Literature-Recommendation-in-Large-scale-Citation-Networks-main')
    processed = Path('data/processed')
    output = processed / 'preliminary_tfidf_paper_84898.csv'
    summary_path = processed / 'preliminary_tfidf_paper_84898_summary.json'
    if output.exists() or summary_path.exists():
        raise FileExistsError('TF-IDF results already exist; preserve prior runs')
    original_paths = [raw / 'article_info.xlsx', raw / 'citation_network.csv',
                      processed / 'preliminary_hybrid_paper_84898.csv',
                      processed / 'preliminary_hybrid_paper_84898_summary.json']
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in original_paths}
    papers = load_papers(raw / 'article_info.xlsx')
    citations = load_citations(raw / 'citation_network.csv')
    counts = validate_data(papers, citations)
    corpus_counts = {'n_articles': counts['articles'],
                     'n_valid_abstracts': counts['articles'] - counts['missing_abstracts'],
                     'n_missing_abstracts': counts['missing_abstracts']}
    print(json.dumps(corpus_counts), flush=True)
    assert corpus_counts == {'n_articles': 220422, 'n_valid_abstracts': 203987,
                             'n_missing_abstracts': 16435}, counts
    assert counts['duplicate_article_ids'] == 0
    start = perf_counter()
    corpus = fit_tfidf(papers)
    diagnostic = {**corpus_counts, 'n_documents': len(corpus.paper_ids),
                  'vocabulary_size': len(corpus.vectorizer.vocabulary_),
                  'matrix_shape': list(corpus.matrix.shape), 'nnz': corpus.matrix.nnz,
                  'dtype': str(corpus.matrix.dtype), 'fit_seconds': perf_counter() - start}
    print(json.dumps(diagnostic), flush=True)
    graph = build_citation_graph(citations)
    plan = prepare_reference_query(papers, graph, 'paper_84898')
    previous_summary = json.loads(original_paths[-1].read_text())
    previous = pd.read_csv(original_paths[-2], float_precision='round_trip').set_index('paper_id')
    assert plan['diagnostic']['n_eligible_candidates'] == 52
    assert plan['diagnostic']['n_candidates_without_abstract'] == 45
    assert plan['excluded'] == {}
    assert sorted(plan['relevant_ids']) == previous_summary['ground_truth']
    eligible = sorted(plan['eligible_ids'])
    assert set(eligible) == set(previous.index)
    for key, value in plan['diagnostic'].items():
        assert previous_summary['counts'][key] == value
    abstracts = tfidf_abstract_similarities(corpus, plan['query_id'], eligible)
    metadata = papers.set_index('paper_id')
    titles = {p: title_similarity(metadata.loc[plan['query_id'], 'title'], metadata.loc[p, 'title'])
              for p in eligible}
    citation_scores = citation_similarities(graph, plan['query_id'], eligible)
    for p in eligible:
        # Exact round-trip CSV floats: any other component change stops comparison.
        assert titles[p] == previous.loc[p, 'title_similarity']
        assert citation_scores[p] == previous.loc[p, 'citation_similarity']
        assert (p in plan['relevant_ids']) == previous.loc[p, 'relevant']
        assert (plan['abstracts'][p] is None) == previous.loc[p, 'abstract_imputed']
    weights = (0.6, 0.2, 0.2)
    assert list(previous_summary['weights'].values()) == list(weights)
    ranking = rank_scores(hybrid_scores(abstracts, titles, citation_scores, weights=weights))
    rows = []
    for rank, (p, score) in enumerate(ranking, 1):
        rows.append({'query_id': plan['query_id'], 'rank': rank, 'paper_id': p,
                     'abstract_similarity': abstracts[p], 'title_similarity': titles[p],
                     'citation_similarity': citation_scores[p], 'weighted_similarity': score,
                     'relevant': p in plan['relevant_ids'],
                     'abstract_imputed': plan['abstracts'][p] is None,
                     'embedding_abstract_similarity': float(previous.loc[p, 'abstract_similarity']),
                     'embedding_rank': int(previous.loc[p, 'rank'])})
    metrics = evaluate_ranking([p for p, _ in ranking], plan['relevant_ids'])
    assert evaluate_ranking(previous.sort_values('rank').index, plan['relevant_ids']) == previous_summary['metrics']
    params = corpus.vectorizer.get_params()
    params['dtype'] = np.dtype(params['dtype']).name
    summary = {'executed_at_utc': datetime.now(timezone.utc).isoformat(),
               'query_id': plan['query_id'], 'label': 'sanity check', 'N': 1,
               'representation': 'TF-IDF', 'candidate_list': 'REFERENCE LIST',
               'weights': previous_summary['weights'], 'counts': plan['diagnostic'],
               'ground_truth': sorted(plan['relevant_ids']), 'n_ranked': len(rows),
               'corpus': diagnostic, 'vectorizer_parameters': params,
               'metrics': metrics, 'embedding_metrics': previous_summary['metrics'],
               'sklearn_version': sklearn.__version__,
               'unchanged_components_verified': True, 'source_sha256': hashes,
               'interpretation': 'Experimental hybrid substitution; N=1 does not establish superiority.',
               'cross_check': [row for row in rows if row['paper_id'] in
                               {'paper_7075', 'paper_14936', 'paper_1314', 'paper_23971', 'paper_25832'}]}
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    pd.DataFrame(rows).to_csv(output, index=False, mode='x')
    with summary_path.open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'metrics': metrics, 'cross_check': summary['cross_check']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
