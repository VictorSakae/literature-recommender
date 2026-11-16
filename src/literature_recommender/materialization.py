"""Stage 7B evidence tables and resumable embedding storage, without ranking."""
from hashlib import sha256
import json
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from literature_recommender.embeddings import embed_texts
from literature_recommender.experiment import prepare_reference_query
from literature_recommender.similarity import citation_similarities, title_similarity

KEYS = ['split', 'query_id', 'candidate_id']
COMMON = KEYS + ['relevant', 'title_similarity', 'citation_similarity', 'candidate_has_abstract']
TEXTUAL = KEYS + ['abstract_similarity', 'abstract_imputed']
MODEL = 'text-embedding-3-small'
DIMENSIONS = 1536


def file_hash(path):
    with Path(path).open('rb') as source:
        return sha256(source.read()).hexdigest()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if file_hash(name) != expected:
            raise ValueError(f'Frozen source/artifact changed: {name}')


def frozen_plans(papers, graph, protocol, population, splits):
    """Reconstruct via the established API, checking every frozen per-query count."""
    expected = dict(seed=42, n_tuning=100, n_evaluation=100,
                    candidate_min=20, candidate_max=None)
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected frozen protocol configuration')
    all_ids = protocol['tuning_query_ids'] + protocol['evaluation_query_ids']
    if len(all_ids) != 200 or len(set(all_ids)) != 200:
        raise ValueError('Frozen queries must be 100+100 disjoint IDs')
    if population.query_id.duplicated().any():
        raise ValueError('Duplicate population IDs')
    if len(population) != protocol['eligible_population_size'] or not population.eligible.all():
        raise ValueError('Frozen population metadata differs')
    metadata = protocol['population_diagnostics']
    if len(papers) != metadata['n_articles'] or len(graph) != metadata['n_graph_nodes']:
        raise ValueError('Corpus/graph metadata differs')
    indexed = population.set_index('query_id')
    plans = []
    for split in ('tuning', 'evaluation'):
        ids = protocol[f'{split}_query_ids']
        table = splits[split]
        if len(ids) != 100 or table.query_id.tolist() != ids:
            raise ValueError(f'Frozen split IDs differ: {split}')
        for row in table.itertuples(index=False):
            q = row.query_id
            if not row.eligible or not indexed.loc[q, 'eligible']:
                raise ValueError(f'Frozen query ineligible: {q}')
            plan = prepare_reference_query(papers, graph, q)
            d = plan['diagnostic']
            counts = {k: d[k] for k in ('n_ground_truth', 'n_structural_candidates',
                      'n_eligible_candidates', 'n_candidates_without_abstract')}
            counts['n_candidates_with_abstract'] = d['n_eligible_candidates'] - d['n_candidates_without_abstract']
            for k, value in counts.items():
                if getattr(row, k) != value or indexed.loc[q, k] != value:
                    raise ValueError(f'Frozen diagnostic differs: {q} {k}')
            fraction = counts['n_candidates_without_abstract'] / counts['n_eligible_candidates']
            if not np.isclose(row.missing_abstract_fraction, fraction, atol=1e-15, rtol=0):
                raise ValueError(f'Frozen missing fraction differs: {q}')
            plans.append((split, plan))
        subset = [p for s, p in plans if s == split]
        if sum(len(p['eligible_ids']) for p in subset) != protocol['diagnostics'][split]['sum_eligible_candidates']:
            raise ValueError('Frozen split pair total differs')
        missing = sum(p['diagnostic']['n_candidates_without_abstract'] for p in subset)
        if missing != protocol['diagnostics'][split]['n_candidates_without_abstract']:
            raise ValueError('Frozen split missing total differs')
    return plans


def common_scores(papers, graph, plans):
    titles = papers.set_index('paper_id').title.to_dict()
    rows = []
    for split, plan in plans:
        q = plan['query_id']
        candidates = sorted(plan['eligible_ids'])
        citations = citation_similarities(graph, q, candidates)
        for p in candidates:
            rows.append((split, q, p, p in plan['relevant_ids'],
                         title_similarity(titles[q], titles[p]), citations[p],
                         plan['abstracts'][p] is not None))
    return pd.DataFrame(rows, columns=COMMON)


def textual_scores(plans, scorer):
    rows = []
    for split, plan in plans:
        q, ids = plan['query_id'], sorted(plan['eligible_ids'])
        scores = scorer(q, ids)
        if set(scores) != set(ids):
            raise ValueError(f'Textual scorer changed candidates for {q}')
        for p in ids:
            rows.append((split, q, p, scores[p], plan['abstracts'][p] is None))
    return pd.DataFrame(rows, columns=TEXTUAL)


def validate_tables(common, representations):
    """Compare actual keys, bool flags, finite bounds and every imputation mean."""
    def keys(frame, columns):
        if list(frame.columns) != columns or frame[KEYS].isna().any().any():
            raise ValueError('Invalid score schema')
        if frame.duplicated(KEYS).any() or frame.query_id.eq(frame.candidate_id).any():
            raise ValueError('Duplicate pair or self candidate')
        if not frame.split.isin(['tuning', 'evaluation']).all():
            raise ValueError('Invalid split')
        if any(not frame[c].map(lambda v: isinstance(v, str) and bool(v)).all() for c in KEYS):
            raise ValueError('Keys must be nonempty strings')
        return frame.set_index(KEYS).sort_index()

    def bounds(values, lower):
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError('Scores must be numeric')
        if not np.isfinite(values).all() or (values < lower - 1e-12).any() or (values > 1 + 1e-12).any():
            raise ValueError('Nonfinite or out-of-range scores')

    base = keys(common, COMMON)
    for column in ['relevant', 'candidate_has_abstract']:
        if common[column].dtype != bool:
            raise ValueError('Common flags must be boolean')
    for column in ['title_similarity', 'citation_similarity']:
        bounds(common[column], 0)
    for name, frame in representations.items():
        indexed = keys(frame, TEXTUAL)
        if not indexed.index.equals(base.index):
            raise ValueError(f'Pair keys differ: {name}')
        if frame.abstract_imputed.dtype != bool:
            raise ValueError('Imputation flags must be boolean')
        if not indexed.abstract_imputed.equals(~base.candidate_has_abstract):
            raise ValueError(f'Imputation flags differ: {name}')
        bounds(frame.abstract_similarity, 0 if name == 'tfidf' else -1)
        for _, group in frame.groupby(['split', 'query_id'], sort=False):
            real = group.loc[~group.abstract_imputed, 'abstract_similarity']
            if real.empty:
                raise ValueError('No real abstract scores')
            mean = fsum(real) / len(real)
            if not np.allclose(group.loc[group.abstract_imputed, 'abstract_similarity'], mean,
                               atol=1e-14, rtol=1e-14):
                raise ValueError(f'Imputation mean differs: {name}')
    return dict(n_pairs=len(common), n_imputed=int((~common.candidate_has_abstract).sum()),
                n_real_scores=int(common.candidate_has_abstract.sum()),
                n_queries=common.groupby('split').query_id.nunique().to_dict())


def needed_texts(plans):
    """Deduplicate by paper identity, rejecting inconsistent text for a shared ID."""
    texts = {}
    for _, plan in plans:
        for p, text in plan['abstracts'].items():
            if text is not None:
                if p in texts and texts[p] != text:
                    raise ValueError(f'Inconsistent text for {p}')
                texts[p] = text
    return dict(sorted(texts.items()))


def _cache_path(directory, paper_id):
    return Path(directory) / (sha256(paper_id.encode()).hexdigest() + '.json')


def cached_vector(directory, paper_id, text):
    path = _cache_path(directory, paper_id)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    identity = dict(paper_id=paper_id, model=MODEL, dimensions=DIMENSIONS,
                    text_sha256=sha256(text.encode()).hexdigest())
    if any(record.get(k) != v for k, v in identity.items()):
        raise ValueError(f'Embedding cache identity mismatch: {paper_id}')
    vector = np.asarray(record['vector'])
    if vector.shape != (DIMENSIONS,) or vector.dtype.kind not in 'iuf' or not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError(f'Invalid cached embedding: {paper_id}')
    return vector


def embedding_inventory(texts, directory):
    available = [p for p, t in texts.items() if cached_vector(directory, p, t) is not None]
    return dict(n_unique_papers_needed=len(texts), n_already_available=len(available),
                n_new_embeddings_required=len(texts) - len(available))


def ensure_embeddings(texts, directory, *, client, batch_size=64):
    """Persist each validated response atomically. Provider failures propagate.

    Completed vectors survive interruption. Interrupted temporary files are ignored.
    Fixed batches of 64 fit the frozen corpus's maximum 4050 UTF-8 bytes/text.
    SDK caller owns timeouts/retries. No failure becomes missing data.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vectors, pending = {}, []
    for p, text in texts.items():
        vector = cached_vector(directory, p, text)
        if vector is None:
            pending.append(p)
        else:
            vectors[p] = vector
    requests = 0
    for start in range(0, len(pending), batch_size):
        ids = pending[start:start + batch_size]
        result = embed_texts([texts[p] for p in ids], client=client)
        requests += 1
        for p, vector in zip(ids, result, strict=True):
            record = dict(paper_id=p, model=MODEL, dimensions=DIMENSIONS,
                          text_sha256=sha256(texts[p].encode()).hexdigest(), vector=vector)
            path = _cache_path(directory, p)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, allow_nan=False))
            temporary.replace(path)
            vectors[p] = vector
        print(f'Embedding batch {requests}: {min(start + batch_size, len(pending))}/{len(pending)} new papers persisted', flush=True)
    return vectors, dict(new_embeddings_generated=len(pending), api_batches=requests)


def persist_scores(frame, path):
    """Reuse identical complete tables, reject any incompatible existing output."""
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path, float_precision='round_trip')
        pd.testing.assert_frame_equal(existing, frame, check_dtype=False, check_exact=True)
        return
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
