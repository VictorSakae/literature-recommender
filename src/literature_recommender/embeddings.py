"""Generate abstract embeddings through a caller-owned OpenAI SDK client."""

from math import isfinite
from numbers import Real

from openai import OpenAI


def embed_texts(texts: list[str], *, client: OpenAI) -> list[list[float]]:
    """Embed texts together using text-embedding-3-small's default 1536 dimensions.

    Return vectors in input order, using provider indices even if data arrives
    reordered. Empty input returns [] without a request. Texts are sent intact;
    reject nonstrings and blank strings before calling the supplied client.
    The caller must omit known missing abstracts, including raw data markers.

    Credentials, client lifetime, timeout and retries belong to the caller.
    Provider errors propagate. Malformed responses raise ValueError, including
    missing/duplicate indices, wrong dimensions and invalid or zero vectors.
    No persistence, truncation, dimensionality override or credential access.
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("Each text must be a string")
        if not text.strip():
            raise ValueError("Texts must not be empty or whitespace-only")
    if not texts:
        return []

    response = client.embeddings.create(
        input=texts.copy(), model="text-embedding-3-small", encoding_format="float",
    )
    data = getattr(response, "data", None)
    if not isinstance(data, list) or len(data) != len(texts):
        raise ValueError("Provider must return one embedding per text")
    by_index: dict[int, list[float]] = {}
    for item in data:
        index = getattr(item, "index", None)
        if (type(index) is not int or not 0 <= index < len(texts)
                or index in by_index):
            raise ValueError("Provider embedding indices must cover input positions once")
        vector = getattr(item, "embedding", None)
        if not isinstance(vector, list) or len(vector) != 1536:
            raise ValueError("Provider embeddings must have 1536 dimensions")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not isfinite(value) for value in vector):
            raise ValueError("Provider embeddings must contain finite real numbers")
        if not any(vector):
            raise ValueError("Provider embeddings must have nonzero norm")
        by_index[index] = [float(value) for value in vector]
    return [by_index[index] for index in range(len(texts))]
