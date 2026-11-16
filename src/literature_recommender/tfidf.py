"""Corpus-fitted sparse abstract representation; experimental hybrid alternative."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from literature_recommender.data import _missing_text


@dataclass
class TfidfCorpus:
    """One shared vocabulary/IDF, with explicit IDs independent of DataFrame index.

    Treat fitted state as read-only. vector() returns a sparse row copy, None
    for known absent abstracts, and raises KeyError for unknown IDs.
    """

    vectorizer: TfidfVectorizer
    paper_ids: tuple[str, ...]
    matrix: csr_matrix
    row_by_paper_id: dict[str, int]
    missing_ids: frozenset[str]

    def vector(self, paper_id: str) -> csr_matrix | None:
        if paper_id in self.missing_ids:
            return None
        return self.matrix[self.row_by_paper_id[paper_id]:self.row_by_paper_id[paper_id] + 1]


def fit_tfidf(papers: pd.DataFrame) -> TfidfCorpus:
    r"""Fit once on ALL valid corpus abstracts, in lexicographic paper-ID order.

    Null, blank and exact 'noabstract' values are absent, never documents.
    Reject duplicate/blank IDs, non-string content and available but tokenless
    abstracts (no silent zero vectors). Preserve text and input data unchanged.
    Text policy: lowercase, Unicode \w+ tokens including single characters,
    accents retained, word unigrams, no linguistic cleanup. See decisions.md.
    """
    ids = papers['paper_id']
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError('Article IDs must be nonblank strings')
    if ids.duplicated().any():
        raise ValueError('Duplicate article IDs make TF-IDF metadata ambiguous')
    abstracts = papers['abstract']
    if any(not isinstance(value, str) for value in abstracts if not pd.isna(value)):
        raise TypeError('Available abstracts must be strings')
    missing = _missing_text(abstracts) | abstracts.eq('noabstract').fillna(False)
    valid = papers.loc[~missing, ['paper_id', 'abstract']].sort_values('paper_id')
    if valid.empty:
        raise ValueError('No valid abstracts to fit TF-IDF')
    vectorizer = TfidfVectorizer(
        input='content', encoding='utf-8', decode_error='strict',
        lowercase=True, strip_accents=None, preprocessor=None, tokenizer=None,
        analyzer='word', token_pattern=r'(?u)\b\w+\b', stop_words=None,
        ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=None,
        vocabulary=None, binary=False, dtype=np.float64, norm='l2',
        use_idf=True, smooth_idf=True, sublinear_tf=False,
    )
    analyze = vectorizer.build_analyzer()
    for paper_id, text in valid.itertuples(index=False, name=None):
        if not analyze(text):
            raise ValueError(f'Abstract for {paper_id!r} must contain word tokens')
    matrix = vectorizer.fit_transform(valid['abstract']).tocsr()
    paper_ids = tuple(valid['paper_id'])
    return TfidfCorpus(vectorizer, paper_ids, matrix,
                       {paper_id: row for row, paper_id in enumerate(paper_ids)},
                       frozenset(ids[missing]))


def tfidf_abstract_similarities(
    corpus: TfidfCorpus, query_id: str, candidate_ids: Iterable[str],
) -> dict[str, float]:
    """Score one eligible list without fitting; impute only known missing text.

    Missing query and all-missing nonempty candidates raise ValueError, as in
    the existing pipeline. Unknown IDs raise KeyError, never become absences.
    Only the 1 x available-candidates cosine result is dense, not the corpus.
    """
    query = corpus.vector(query_id)
    if query is None:
        raise ValueError('Query must have an available abstract')
    candidates = list(dict.fromkeys(candidate_ids))
    if not candidates:
        return {}
    available = []
    for paper_id in candidates:
        if paper_id not in corpus.missing_ids:
            # Explicit lookup distinguishes unknown IDs from missing abstracts.
            available.append((paper_id, corpus.row_by_paper_id[paper_id]))
    if not available:
        raise ValueError('No candidate abstracts available for a mean')
    similarities = cosine_similarity(query, corpus.matrix[[row for _, row in available]])[0]
    scores = {paper_id: float(score) for (paper_id, _), score in zip(available, similarities)}
    mean = fsum(scores.values()) / len(scores)
    return {paper_id: scores.get(paper_id, mean) for paper_id in candidates}
