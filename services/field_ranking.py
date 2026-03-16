"""
Field-weighted BM25 scorer for multi-field legal documents.

Extends the base BM25 algorithm by scoring each document field
(title, citation, headnote, text) independently and combining the
per-field scores with configurable boost weights.  This produces more
relevant results than concatenating all fields into a single string,
because it:

- Prevents long *text* fields from drowning out short but highly
  relevant *title* or *citation* matches.
- Allows practitioners to tune which fields matter most (e.g. a
  citation match is nearly always a high-precision hit).
- Preserves BM25 length-normalisation *within* each field, so IDF and
  TF saturation behave correctly per field.

Usage::

    from services.field_ranking import FieldBM25Scorer, FieldWeights

    docs = [
        {"title": "Smith v Jones", "citation": "2021 SC 12",
         "headnote": "Contract breach", "text": "...full text..."},
        ...
    ]

    weights = FieldWeights(title=3.0, citation=4.0, headnote=2.0, text=1.0)
    scorer = FieldBM25Scorer(docs, weights=weights)
    ranked = scorer.rank("Smith contract")
    # → [(doc_idx, combined_score), ...] sorted best-first

Algorithm
---------
For each field f with boost w_f::

    field_score(q, d, f) = BM25(q, d_f)   # standard Okapi BM25 within field f

    combined(q, d) = sum_f [ w_f * field_score(q, d, f) ]

BM25 parameters (k1, b) are shared across all fields.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# Reuse the same stop-word list from the base ranker.
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

# Ordered list of supported field names.
_FIELDS: tuple[str, ...] = ("title", "citation", "headnote", "text")


@dataclass
class FieldWeights:
    """Per-field BM25 boost multipliers.

    All weights must be non-negative.  A weight of ``0.0`` disables the
    field entirely.  Weights are *not* normalised, so setting
    ``title=3.0`` means title matches contribute three times more than a
    ``text`` field with ``weight=1.0``.

    Defaults are tuned for legal case-law search where citations and
    titles carry high precision signal.
    """

    title: float = 3.0
    citation: float = 4.0
    headnote: float = 2.0
    text: float = 1.0

    def __post_init__(self) -> None:
        for field in _FIELDS:
            val = getattr(self, field)
            if val < 0:
                raise ValueError(f"FieldWeights.{field} must be >= 0, got {val!r}")

    def as_dict(self) -> dict[str, float]:
        return {f: getattr(self, f) for f in _FIELDS}


class _FieldIndex:
    """BM25 index for a single field across the whole corpus.

    Parameters
    ----------
    field_texts:
        One string per document (the value of a single field, e.g. ``title``).
        May contain empty strings for documents that lack that field.
    k1:
        TF saturation parameter.
    b:
        Length normalisation parameter.
    """

    def __init__(
        self,
        field_texts: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._n = len(field_texts)

        # Tokenise every document in the field
        self._corpus: list[list[str]] = [_tokenize(t) for t in field_texts]

        total = sum(len(toks) for toks in self._corpus)
        self._avgdl: float = total / self._n if self._n else 0.0

        # Per-document TF
        self._tf: list[Counter[str]] = [Counter(toks) for toks in self._corpus]

        # Document frequency per term
        self._df: dict[str, int] = {}
        for tf_c in self._tf:
            for term in tf_c:
                self._df[term] = self._df.get(term, 0) + 1

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        """BM25 score for *query_terms* against document *doc_idx*."""
        if not query_terms or self._n == 0:
            return 0.0

        doc_len = len(self._corpus[doc_idx])
        tf_counter = self._tf[doc_idx]
        length_norm = (1 - self.b + self.b * doc_len / self._avgdl) if self._avgdl > 0 else 1.0

        total = 0.0
        for term in query_terms:
            if term not in tf_counter:
                continue
            tf = tf_counter[term]
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * length_norm
            total += idf * (numerator / denominator)

        return total

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1)


class FieldBM25Scorer:
    """Multi-field BM25 scorer that combines per-field scores with boosts.

    Parameters
    ----------
    documents:
        List of dicts with keys ``"title"``, ``"citation"``,
        ``"headnote"``, ``"text"``.  Missing keys default to ``""``.
    weights:
        Per-field boost multipliers.  See :class:`FieldWeights`.
    k1:
        BM25 TF saturation parameter (default ``1.5``).
    b:
        BM25 length normalisation parameter (default ``0.75``).
    """

    def __init__(
        self,
        documents: list[dict[str, str]],
        *,
        weights: FieldWeights | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b!r}")
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1!r}")

        self._weights = weights or FieldWeights()
        self._n = len(documents)

        # Build a per-field BM25 index
        self._indexes: dict[str, _FieldIndex] = {
            field: _FieldIndex(
                [doc.get(field, "") or "" for doc in documents],
                k1=k1,
                b=b,
            )
            for field in _FIELDS
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, query: str, doc_idx: int) -> float:
        """Return the combined field-weighted BM25 score.

        Parameters
        ----------
        query:
            Raw query string.
        doc_idx:
            Zero-based document index.

        Returns
        -------
        float
            Non-negative combined score.

        Raises
        ------
        IndexError
            If *doc_idx* is out of range.
        """
        if doc_idx < 0 or doc_idx >= self._n:
            raise IndexError(f"doc_idx {doc_idx} out of range for corpus of size {self._n}")

        query_terms = _tokenize(query)
        if not query_terms:
            return 0.0

        weight_dict = self._weights.as_dict()
        total = 0.0
        for field, index in self._indexes.items():
            w = weight_dict[field]
            if w > 0:
                total += w * index.score(query_terms, doc_idx)
        return total

    def rank(self, query: str) -> list[tuple[int, float]]:
        """Rank all documents by combined field-weighted BM25 score.

        Returns
        -------
        list[tuple[int, float]]
            ``(doc_idx, score)`` pairs, sorted descending by score.
            Documents scoring ``0`` are excluded.
        """
        scored = ((i, self.score(query, i)) for i in range(self._n))
        return sorted(
            ((i, s) for i, s in scored if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )

    @property
    def num_documents(self) -> int:
        """Number of documents in the corpus."""
        return self._n


# ------------------------------------------------------------------
# Convenience helpers
# ------------------------------------------------------------------


def score_cases(
    query: str,
    cases: list,
    *,
    weights: FieldWeights | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Score a list of ORM ``Case`` objects with field-weighted BM25.

    Extracts ``title``, ``citation``, ``headnote``, and ``text`` attributes
    from each case object and returns a normalised ``[0, 1]`` score list.

    Parameters
    ----------
    query:
        Search query string.
    cases:
        Iterable of objects with ``title``, ``citation``, ``headnote``,
        and ``text`` attributes.
    weights:
        Field boost weights.  Defaults to :class:`FieldWeights` defaults.
    k1, b:
        BM25 tuning parameters.

    Returns
    -------
    list[float]
        Normalised relevance scores in ``[0.0, 1.0]``, same order as *cases*.
    """
    if not cases:
        return []
    if not query:
        return [1.0] * len(cases)

    docs = [
        {
            "title": getattr(c, "title", "") or "",
            "citation": getattr(c, "citation", "") or "",
            "headnote": getattr(c, "headnote", "") or "",
            "text": getattr(c, "text", "") or "",
        }
        for c in cases
    ]

    scorer = FieldBM25Scorer(docs, weights=weights, k1=k1, b=b)
    raw = [scorer.score(query, i) for i in range(len(cases))]

    max_score = max(raw) if any(s > 0 for s in raw) else 1.0
    if max_score == 0:
        return [0.0] * len(cases)
    return [s / max_score for s in raw]


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, remove stop-words and singles."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
