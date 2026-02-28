"""
BM25 relevance scorer for re-ranking search results.

Implements the Okapi BM25 ranking function (Robertson & Walker 1994):

    score(q, d) = sum_t [ IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl)) ]

where:
    tf(t, d)   = term frequency of t in document d
    IDF(t)     = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)   (Robertson smoothing)
    |d|        = document length (number of tokens)
    avgdl      = mean document length across the corpus
    k1, b      = tuning parameters (defaults: k1=1.5, b=0.75)

Typical usage::

    texts = [case.title + " " + (case.headnote or "") for case in cases]
    scorer = BM25Scorer(texts)
    scores = [scorer.score(query, i) for i in range(len(texts))]
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Generic English stop-words to strip before indexing.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


class BM25Scorer:
    """Okapi BM25 relevance scorer over a fixed corpus of text documents.

    Build a scorer once for a set of documents, then call :meth:`score` or
    :meth:`rank` to get relevance values for any query string.

    Parameters
    ----------
    documents:
        List of raw text strings that form the corpus.  Documents are
        tokenised at construction time; the original strings are not kept.
    k1:
        Term-frequency saturation parameter (default ``1.5``).
        Higher values reduce the diminishing-returns effect on repeated terms.
    b:
        Length normalisation parameter (default ``0.75``).
        ``b=0`` disables length normalisation; ``b=1`` applies full normalisation.
    """

    def __init__(
        self,
        documents: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b!r}")
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1!r}")

        self.k1 = k1
        self.b = b

        self._corpus: list[list[str]] = [self._tokenize(d) for d in documents]
        self._n: int = len(self._corpus)

        total_tokens = sum(len(tokens) for tokens in self._corpus)
        self._avgdl: float = total_tokens / self._n if self._n > 0 else 0.0

        # Per-document term frequencies
        self._tf: list[Counter[str]] = [Counter(tokens) for tokens in self._corpus]

        # Document frequency: how many docs contain each term
        self._df: dict[str, int] = {}
        for tf_counter in self._tf:
            for term in tf_counter:
                self._df[term] = self._df.get(term, 0) + 1

    # Public API

    def score(self, query: str, doc_idx: int) -> float:
        """Return the BM25 relevance score for *query* against document *doc_idx*.

        Parameters
        ----------
        query:
            Raw query string (tokenised the same way as documents).
        doc_idx:
            Zero-based index into the corpus supplied at construction.

        Returns
        -------
        float
            Non-negative BM25 score.  Returns ``0.0`` when the query is empty
            or no query terms appear in the document.

        Raises
        ------
        IndexError
            When *doc_idx* is outside ``[0, len(documents))``.
        """
        if doc_idx < 0 or doc_idx >= self._n:
            raise IndexError(f"doc_idx {doc_idx} out of range for corpus of size {self._n}")

        query_terms = self._tokenize(query)
        if not query_terms:
            return 0.0

        doc_len = len(self._corpus[doc_idx])
        tf_counter = self._tf[doc_idx]

        # Avoid division-by-zero when avgdl is 0 (empty corpus edge case)
        length_norm = (1 - self.b + self.b * doc_len / self._avgdl) if self._avgdl > 0 else 1.0

        result = 0.0
        for term in query_terms:
            if term not in tf_counter:
                continue
            tf = tf_counter[term]
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * length_norm
            result += idf * (numerator / denominator)

        return result

    def rank(self, query: str) -> list[tuple[int, float]]:
        """Rank all corpus documents by BM25 relevance for *query*.

        Parameters
        ----------
        query:
            Raw query string.

        Returns
        -------
        list[tuple[int, float]]
            ``(doc_idx, score)`` pairs sorted by score descending.
            Documents that score ``0`` are excluded.
        """
        scored = ((idx, self.score(query, idx)) for idx in range(self._n))
        return sorted(
            ((idx, s) for idx, s in scored if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )

    # Static helpers

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, extract alphanumeric tokens, strip stop-words.

        Single-character tokens are also removed to avoid noise from
        punctuation fragments.
        """
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    def _idf(self, term: str) -> float:
        """Robertson-smoothed IDF -- always non-negative.

            IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        A term present in every document still receives a small positive weight.
        """
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1)


def score_results(
    query: str,
    texts: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Convenience wrapper: return BM25 scores for each text in *texts*.

    Parameters
    ----------
    query:
        Search query string.
    texts:
        Corpus documents -- one string per result.
    k1, b:
        BM25 tuning parameters (see :class:`BM25Scorer`).

    Returns
    -------
    list[float]
        BM25 score for each document, in the same order as *texts*.
    """
    if not texts:
        return []
    scorer = BM25Scorer(texts, k1=k1, b=b)
    return [scorer.score(query, i) for i in range(len(texts))]
