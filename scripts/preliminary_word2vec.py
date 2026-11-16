"""Stage 6B: frozen N=1 check. Run with PYTHONHASHSEED=42 through uv."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import gensim
import numpy as np
import pandas as pd

from literature_recommender.data import load_papers, load_citations
from literature_recommender.evaluation import evaluate_ranking
from literature_recommender.experiment import prepare_reference_query
from literature_recommender.graph import build_citation_graph
from literature_recommender.recommender import hybrid_scores, rank_scores
from literature_recommender.similarity import title_similarity, citation_similarities
from literature_recommender.word2vec import (
    WORD2VEC_PARAMETERS, prepare_word2vec_corpus, tokenize_abstract,
    train_word2vec, word2vec_abstract_similarities,
)


def main():
    if os.environ.get('PYTHONHASHSEED') != '42':
        raise ValueError('Start Python with PYTHONHASHSEED=42')
    processed = Path('data/processed')
    raw = Path('data/raw/Academic-Literature-Recommendation-in-Large-scale-Citation-Networks-main')
    output = processed / 'preliminary_word2vec_paper_84898.csv'
    summary_path = processed / 'preliminary_word2vec_paper_84898_summary.json'
    model_path = processed / 'models/word2vec_cbow_100_seed42.model'
    if any(p.exists() for p in (output, summary_path, model_path)):
        raise FileExistsError('Preserve existing Word2Vec results/model')
    sources = [raw / 'article_info.xlsx', raw / 'citation_network.csv'] + [
        processed / f'preliminary_{rep}_paper_84898{suffix}'
        for rep in ('hybrid', 'tfidf') for suffix in ('.csv', '_summary.json')]
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in sources}
    papers = load_papers(sources[0])
    corpus = prepare_word2vec_corpus(papers)
    counts = dict(n_articles=len(papers), n_valid_abstracts=len(corpus.texts),
                  n_missing_abstracts=len(corpus.missing_ids))
    assert counts == dict(n_articles=220422, n_valid_abstracts=203987, n_missing_abstracts=16435)
    total_tokens = sum(len(tokens) for tokens in corpus)
    print(json.dumps({**counts, 'n_total_tokens': total_tokens,
                      'parameters': WORD2VEC_PARAMETERS}), flush=True)
    # Freeze and verify all controls BEFORE training or comparison.
    graph = build_citation_graph(load_citations(sources[1]))
    plan = prepare_reference_query(papers, graph, 'paper_84898')
    eligible = sorted(plan['eligible_ids'])
    metadata = papers.set_index('paper_id')
    titles = {p: title_similarity(metadata.loc['paper_84898', 'title'], metadata.loc[p, 'title'])
              for p in eligible}
    citations = citation_similarities(graph, 'paper_84898', eligible)
    controls = {}
    weights = (0.6, 0.2, 0.2)
    for rep in ('hybrid', 'tfidf'):
        frame = pd.read_csv(processed / f'preliminary_{rep}_paper_84898.csv',
                            float_precision='round_trip').set_index('paper_id')
        summary = json.loads((processed / f'preliminary_{rep}_paper_84898_summary.json').read_text())
        assert set(frame.index) == set(eligible) == plan['candidate_ids']
        assert plan['excluded'] == {}
        for key, value in plan['diagnostic'].items():
            assert summary['counts'][key] == value
        if 'n_excluded_candidates' in summary['counts']:
            assert summary['counts']['n_excluded_candidates'] == len(plan['excluded'])
            assert summary['counts']['n_candidates_with_abstract'] == sum(
                p not in corpus.missing_ids for p in eligible)
        assert sorted(plan['relevant_ids']) == summary['ground_truth']
        assert [summary['weights'][k] for k in ('abstract', 'title', 'citation')] == list(weights)
        for p in eligible:
            assert titles[p] == frame.loc[p, 'title_similarity']
            assert citations[p] == frame.loc[p, 'citation_similarity']
            assert (p in plan['relevant_ids']) == frame.loc[p, 'relevant']
            assert (p in corpus.missing_ids) == frame.loc[p, 'abstract_imputed']
        old_scores = hybrid_scores(frame.abstract_similarity.to_dict(), titles, citations)
        assert [p for p, _ in rank_scores(old_scores)] == list(frame.sort_values('rank').index)
        assert evaluate_ranking(frame.sort_values('rank').index, plan['relevant_ids']) == summary['metrics']
        controls[rep] = (frame, summary)
    assert len(eligible) == 52 and len(plan['relevant_ids']) == 5
    print('Controls verified against both prior representations; starting fixed training.', flush=True)
    start = perf_counter()
    model = train_word2vec(corpus)
    training_seconds = perf_counter() - start
    diagnostic = {**counts, 'n_total_tokens': total_tokens, 'vocabulary_size': len(model.wv),
                  'vector_size': model.vector_size, 'epochs': model.epochs,
                  'corpus_count': model.corpus_count, 'corpus_total_words': model.corpus_total_words,
                  'training_seconds': training_seconds}
    assert model.corpus_count == counts['n_valid_abstracts']
    assert model.corpus_total_words == total_tokens
    # One streaming diagnostic pass; no corpus document-vector matrix stored.
    no_representation = []
    n_in_vocab = 0
    for p, text in corpus.texts.items():
        tokens = tokenize_abstract(text)
        represented = sum(t in model.wv for t in tokens)
        n_in_vocab += represented
        if not represented:
            no_representation.append(p)
    diagnostic.update(n_tokens_in_vocab=n_in_vocab,
                      n_valid_abstracts_without_representation=len(no_representation),
                      valid_abstracts_without_representation=no_representation)
    print(json.dumps(diagnostic), flush=True)
    # Confirm ignore rule and estimate size before writing a single native model.
    import subprocess
    subprocess.run(['git', 'check-ignore', '--no-index', str(model_path)], check=True)
    estimate = sum(v.nbytes for v in (model.wv.vectors, model.syn1neg))
    print(f'Model vector arrays approximately {estimate} bytes; ignored directory: {model_path.parent}', flush=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), separately=[])
    abstracts = word2vec_abstract_similarities(corpus, model.wv, 'paper_84898', eligible)
    ranking = rank_scores(hybrid_scores(abstracts, titles, citations, weights=weights))
    rows = []
    for rank, (p, score) in enumerate(ranking, 1):
        tokens = [] if p in corpus.missing_ids else tokenize_abstract(corpus.texts[p])
        row = dict(query_id='paper_84898', paper_id=p, rank=rank,
                   abstract_similarity=abstracts[p], title_similarity=titles[p],
                   citation_similarity=citations[p], weighted_similarity=score,
                   relevant=p in plan['relevant_ids'], abstract_imputed=p in corpus.missing_ids,
                   n_tokens_total=len(tokens) if tokens else None,
                   n_tokens_in_vocab=sum(t in model.wv for t in tokens) if tokens else None)
        for rep, name in (('hybrid', 'embedding'), ('tfidf', 'tfidf')):
            frame = controls[rep][0]
            row[f'{name}_abstract_similarity'] = float(frame.loc[p, 'abstract_similarity'])
            row[f'{name}_rank'] = int(frame.loc[p, 'rank'])
        rows.append(row)
    metrics = evaluate_ranking([p for p, _ in ranking], plan['relevant_ids'])
    summary = dict(executed_at_utc=datetime.now(timezone.utc).isoformat(), query_id='paper_84898',
                   label='sanity check', N=1, tuning=False, representation='Word2Vec',
                   candidate_list='REFERENCE LIST', weights=dict(zip(('abstract', 'title', 'citation'), weights)),
                   counts=plan['diagnostic'], ground_truth=sorted(plan['relevant_ids']), n_ranked=len(rows),
                   corpus=diagnostic, parameters=WORD2VEC_PARAMETERS, gensim_version=gensim.__version__,
                   numpy_version=np.__version__, pythonhashseed=os.environ['PYTHONHASHSEED'],
                   aggregation='arithmetic mean of in-vocabulary token occurrences; no normalization',
                   token_pattern=r'(?u)\b\w+\b', lowercase=True,
                   missing_policy='same eligible list mean; all-OOV raises ValueError',
                   model_path=str(model_path), model_bytes=model_path.stat().st_size,
                   metrics=metrics, embedding_metrics=controls['hybrid'][1]['metrics'],
                   tfidf_metrics=controls['tfidf'][1]['metrics'], unchanged_components_verified=True,
                   source_sha256=hashes,
                   cross_check=[r for r in rows if r['paper_id'] in
                                {'paper_7075', 'paper_25832', 'paper_1510', 'paper_1314',
                                 'paper_27175', 'paper_14936', 'paper_23971'}])
    assert all(sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    pd.DataFrame(rows).to_csv(output, index=False, mode='x')
    with summary_path.open('x') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(metrics=metrics, cross_check=summary['cross_check']), indent=2), flush=True)


if __name__ == '__main__':
    main()
