"""Global Word2Vec training and mean abstract vectors (experimental hybrid).

Train once, then reuse model.wv. Parameters are reproduction choices, not
claimed author settings. See docs/decisions.md for evidence and determinism.
"""

from collections.abc import Iterable
from dataclasses import dataclass
import re

from gensim.models import KeyedVectors, Word2Vec
import numpy as np
import pandas as pd

from literature_recommender.data import _missing_text
from literature_recommender.similarity import abstract_similarities

WORD2VEC_PARAMETERS = dict(
    sg=0, vector_size=100, window=5, min_count=5, workers=1, epochs=5,
    negative=5, hs=0, sample=0.001, seed=42, alpha=0.025,
    min_alpha=0.0001, cbow_mean=1, ns_exponent=0.75,
    sorted_vocab=1, shrink_windows=True, batch_words=10000,
    max_vocab_size=None, max_final_vocab=None, null_word=0,
)
_WORD = re.compile(r'(?u)\b\w+\b')


def tokenize_abstract(text: str) -> list[str]:
    """TF-IDF-compatible Unicode words, lowercase, retaining accents and digits."""
    if not isinstance(text, str):
        raise TypeError('Abstract must be a string')
    return _WORD.findall(text.lower())


@dataclass
class Word2VecCorpus:
    """Reiterable token stream, retaining texts rather than all token lists.

    Construct with prepare_word2vec_corpus; treat state as read-only. Missing
    IDs are distinct from valid texts whose tokens are outside a model's vocab.
    """

    texts: dict[str, str]
    missing_ids: frozenset[str]

    def __iter__(self):
        for text in self.texts.values():
            yield tokenize_abstract(text)

    def vector(self, paper_id: str, word_vectors: KeyedVectors) -> np.ndarray | None:
        """Return an independent mean vector, None only for known missing text."""
        if paper_id in self.missing_ids:
            return None
        return document_vector(tokenize_abstract(self.texts[paper_id]), word_vectors)


def prepare_word2vec_corpus(papers: pd.DataFrame) -> Word2VecCorpus:
    """Validate without mutation; keep all valid abstracts sorted by paper ID.

    Repeated texts for distinct papers are retained. Null, blank and exact
    noabstract are omitted. Available tokenless text is an error, not absence.
    """
    ids, abstracts = papers['paper_id'], papers['abstract']
    if any(not isinstance(p, str) or not p.strip() for p in ids):
        raise ValueError('Article IDs must be nonblank strings')
    if ids.duplicated().any():
        raise ValueError('Duplicate article IDs make Word2Vec metadata ambiguous')
    if any(not isinstance(t, str) for t in abstracts if not pd.isna(t)):
        raise TypeError('Available abstracts must be strings')
    missing = _missing_text(abstracts) | abstracts.eq('noabstract').fillna(False)
    texts = dict(papers.loc[~missing, ['paper_id', 'abstract']]
                 .sort_values('paper_id').itertuples(index=False, name=None))
    if not texts:
        raise ValueError('No valid abstracts to train Word2Vec')
    for p, text in texts.items():
        if not tokenize_abstract(text):
            raise ValueError(f'Abstract for {p!r} must contain word tokens')
    return Word2VecCorpus(texts, frozenset(ids[missing]))


def train_word2vec(corpus: Word2VecCorpus) -> Word2Vec:
    """One fixed CBOW fit on the whole corpus, never inside list scoring.

    Run with PYTHONHASHSEED=42 and the locked environment for repeatability;
    workers=1 avoids asynchronous training. No cross-platform bitwise guarantee.
    """
    model = Word2Vec(**WORD2VEC_PARAMETERS)
    model.build_vocab(corpus)
    if not len(model.wv):
        raise ValueError('No tokens meet Word2Vec min_count')
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def document_vector(tokens: Iterable[str], word_vectors: KeyedVectors) -> np.ndarray:
    """Arithmetic mean over in-vocabulary occurrences; ignore OOV, no L2 step.

    Raise for zero representable tokens or an invalid/zero mean; never impute
    technical representation failures as missing abstracts. Inputs are unchanged.
    """
    vectors = [word_vectors[t] for t in tokens if t in word_vectors]
    if not vectors:
        raise ValueError('Abstract has no tokens in the Word2Vec vocabulary')
    mean = np.mean(vectors, axis=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.any(mean):
        raise ValueError('Abstract mean vector must be finite and nonzero')
    return mean


def word2vec_abstract_similarities(
    corpus: Word2VecCorpus, word_vectors: KeyedVectors,
    query_id: str, candidate_ids: Iterable[str],
) -> dict[str, float]:
    """Existing cosine and list-local missing-abstract policy; no training."""
    query = corpus.vector(query_id, word_vectors)
    if query is None:
        raise ValueError('Query must have an available abstract')
    candidates = {p: corpus.vector(p, word_vectors) for p in dict.fromkeys(candidate_ids)}
    return abstract_similarities(query, candidates)
