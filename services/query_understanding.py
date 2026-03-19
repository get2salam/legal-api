"""
Query understanding module for legal search.

Parses, normalises, and enriches user queries before they reach the
search engine.  The pipeline has four stages:

1. **Normalisation** — lowercase, Unicode normalisation, whitespace
   cleanup, removal of punctuation noise.
2. **Intent classification** — lightweight heuristic classification into
   one of several ``QueryIntent`` values so callers can route queries
   differently (e.g. citation lookups bypass full-text BM25).
3. **Entity extraction** — pull structured entities from free text:
   citations, court names, year constraints, and quoted phrases.
4. **Expansion** — return a list of synonym terms the caller *may* inject
   into the search; this is opt-in so callers control whether expansion
   happens.

No ML model is required: everything runs on regex + trie lookups so
latency stays sub-millisecond even for large corpora.

Usage::

    from services.query_understanding import understand_query

    result = understand_query("contract breach Supreme Court 2021")
    # QueryResult(
    #     normalised="contract breach supreme court 2021",
    #     intent=<QueryIntent.TOPICAL: 'topical'>,
    #     entities=QueryEntities(courts=['supreme court'], years=[2021], ...),
    #     expansions=['agreement', 'obligation', 'damages'],
    # )
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

# ---------------------------------------------------------------------------
# Public API types
# ---------------------------------------------------------------------------


class QueryIntent(StrEnum):
    """Coarse intent label for a user query.

    ``CITATION``
        Query contains a structured citation reference (year + court code +
        case number).  High-precision lookup — skip BM25, do exact match.

    ``TOPICAL``
        General subject-matter search (most queries fall here).

    ``JUDGE``
        Query appears to target a specific judge by name.

    ``COURT``
        Query primarily filters by court name with minimal keyword content.

    ``DATE_RANGE``
        Query expresses a year range (e.g. "between 2018 and 2022").

    ``UNKNOWN``
        Could not determine intent with reasonable confidence.
    """

    CITATION = "citation"
    TOPICAL = "topical"
    JUDGE = "judge"
    COURT = "court"
    DATE_RANGE = "date_range"
    UNKNOWN = "unknown"


@dataclass
class QueryEntities:
    """Structured entities extracted from a query string.

    Attributes
    ----------
    citations : list[str]
        Detected citation patterns, e.g. ``["2021 SC 45"]``.
    courts : list[str]
        Matched court name tokens, e.g. ``["supreme court"]``.
    years : list[int]
        Four-digit years found in the query, e.g. ``[2021, 2022]``.
    year_range : tuple[int, int] | None
        ``(from_year, to_year)`` when "between X and Y" phrasing is found.
    quoted_phrases : list[str]
        Text inside double quotes, e.g. ``["breach of trust"]``.
    judge_names : list[str]
        Names that follow judge title keywords, e.g. ``["Justice Khan"]``.
    """

    citations: list[str] = field(default_factory=list)
    courts: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    year_range: tuple[int, int] | None = None
    quoted_phrases: list[str] = field(default_factory=list)
    judge_names: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """Full output from the query understanding pipeline.

    Attributes
    ----------
    original : str
        The raw input string.
    normalised : str
        Cleaned query ready to pass to the search engine.
    intent : QueryIntent
        Detected intent label.
    entities : QueryEntities
        Structured entities extracted from the query.
    expansions : list[str]
        Optional synonym terms the caller may add to broaden the search.
    tokens : list[str]
        Non-stop-word tokens remaining after normalisation.
    """

    original: str
    normalised: str
    intent: QueryIntent
    entities: QueryEntities
    expansions: list[str]
    tokens: list[str]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Citation patterns: "YYYY SC 123", "YYYY MLD 45", "YYYY CLC 789", etc.
_CITATION_RE: Final = re.compile(
    r"\b(\d{4})\s+"
    r"(SC|MLD|CLC|SCMR|PLD|PCrLJ|YLR|SBLR|PTD|ACR|NLR)\s+"
    r"(\d+)\b",
    re.IGNORECASE,
)

# Bare year: four digits in legal range
_YEAR_RE: Final = re.compile(r"\b(1[5-9]\d{2}|20[0-2]\d)\b")

# Year range: "between 2018 and 2022" or "2018 to 2022" or "2018-2022"
_YEAR_RANGE_RE: Final = re.compile(
    r"\b(?:between\s+)?(\d{4})\s*(?:to|and|[-–—])\s*(\d{4})\b",
    re.IGNORECASE,
)

# Quoted phrases
_QUOTED_RE: Final = re.compile(r'"([^"]+)"')

# Judge title keywords
_JUDGE_TITLE_RE: Final = re.compile(
    r"\b(Justice|Judge|CJ|Chief\s+Justice|Honourable|Hon\.?)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
    re.IGNORECASE,
)

# Known court names (lowercase canonical → display form)
_COURT_NAMES: Final[dict[str, str]] = {
    "supreme court": "Supreme Court",
    "sc": "Supreme Court",
    "high court": "High Court",
    "hc": "High Court",
    "federal shariat court": "Federal Shariat Court",
    "fsc": "Federal Shariat Court",
    "lahore high court": "Lahore High Court",
    "lhc": "Lahore High Court",
    "sindh high court": "Sindh High Court",
    "shc": "Sindh High Court",
    "islamabad high court": "Islamabad High Court",
    "ihc": "Islamabad High Court",
    "peshawar high court": "Peshawar High Court",
    "phc": "Peshawar High Court",
    "balochistan high court": "Balochistan High Court",
    "bhc": "Balochistan High Court",
    "appellate tribunal": "Appellate Tribunal",
    "district court": "District Court",
    "sessions court": "Sessions Court",
}

# Subject synonym groups  (each inner list is a synonym cluster)
_SYNONYM_GROUPS: Final[list[list[str]]] = [
    ["contract", "agreement", "deed", "obligation", "covenant"],
    ["breach", "violation", "infringement", "contravention", "non-compliance"],
    ["negligence", "carelessness", "recklessness", "duty of care"],
    ["fraud", "misrepresentation", "deceit", "dishonesty", "forgery"],
    ["murder", "homicide", "culpable homicide", "manslaughter", "killing"],
    ["property", "land", "estate", "immovable", "title", "ownership"],
    ["custody", "guardianship", "welfare", "minor", "child"],
    ["bail", "surety", "bond", "recognizance", "pre-arrest bail"],
    ["dismissal", "termination", "removal", "discharge", "reinstatement"],
    ["damages", "compensation", "remedy", "relief", "restitution"],
    ["appeal", "revision", "review", "petition", "writ"],
    ["evidence", "testimony", "witness", "proof", "corroboration"],
    ["sentence", "punishment", "penalty", "imprisonment", "fine"],
    ["injunction", "stay", "restraint", "interlocutory", "interim relief"],
    ["constitution", "fundamental rights", "constitutional", "article 9", "article 10"],
    ["tax", "taxation", "income tax", "customs", "levy", "duty"],
    ["defamation", "libel", "slander", "reputation", "honor"],
    ["divorce", "dissolution", "talaq", "khula", "separation"],
    ["succession", "inheritance", "will", "testamentary", "estate"],
    ["contempt", "disobedience", "obstruction", "interference"],
]

# Build term → cluster index lookup
_TERM_TO_CLUSTER: Final[dict[str, int]] = {}
for _ci, _group in enumerate(_SYNONYM_GROUPS):
    for _term in _group:
        for _word in _term.split():
            _TERM_TO_CLUSTER[_word.lower()] = _ci

# Stop-words to strip from the normalised query
_STOP_WORDS: Final[frozenset[str]] = frozenset(
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
        "vs",
        "versus",
    }
)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def understand_query(raw: str) -> QueryResult:
    """Run the full query understanding pipeline.

    Parameters
    ----------
    raw:
        Raw user input string.

    Returns
    -------
    QueryResult
        Normalised query, intent, entities, and expansion hints.
    """
    if not raw or not raw.strip():
        return QueryResult(
            original=raw or "",
            normalised="",
            intent=QueryIntent.UNKNOWN,
            entities=QueryEntities(),
            expansions=[],
            tokens=[],
        )

    normalised = _normalise(raw)
    entities = _extract_entities(raw, normalised)
    intent = _classify_intent(normalised, entities)
    tokens = _tokenise(normalised)
    expansions = _build_expansions(tokens, entities.quoted_phrases)

    return QueryResult(
        original=raw,
        normalised=normalised,
        intent=intent,
        entities=entities,
        expansions=expansions,
        tokens=tokens,
    )


# ---------------------------------------------------------------------------
# Stage 1: Normalisation
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Return a cleaned, lowercase query string.

    Steps:
    - Unicode NFC normalisation (combines diacritics).
    - Lowercase.
    - Collapse runs of whitespace/newlines to single space.
    - Strip leading/trailing whitespace.
    - Remove characters that are not alphanumeric, space, hyphen, or quote.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Preserve hyphens and double-quotes; strip everything else non-alnum
    text = re.sub(r'[^\w\s\-"]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 2: Entity extraction
# ---------------------------------------------------------------------------


def _extract_entities(raw: str, normalised: str) -> QueryEntities:
    """Extract structured entities from the raw (and normalised) query."""
    entities = QueryEntities()

    # -- Citations -----------------------------------------------------------
    for m in _CITATION_RE.finditer(raw):
        entities.citations.append(m.group(0).strip())

    # -- Quoted phrases ------------------------------------------------------
    for m in _QUOTED_RE.finditer(raw):
        phrase = m.group(1).strip()
        if phrase:
            entities.quoted_phrases.append(phrase)

    # -- Judge names ---------------------------------------------------------
    for m in _JUDGE_TITLE_RE.finditer(raw):
        name = f"{m.group(1)} {m.group(2)}".strip()
        entities.judge_names.append(name)

    # -- Year range (must run before bare year extraction) -------------------
    range_match = _YEAR_RANGE_RE.search(normalised)
    if range_match:
        y1, y2 = int(range_match.group(1)), int(range_match.group(2))
        entities.year_range = (min(y1, y2), max(y1, y2))

    # -- Bare years ----------------------------------------------------------
    for m in _YEAR_RE.finditer(raw):
        yr = int(m.group(1))
        entities.years.append(yr)
    # Deduplicate while preserving order
    seen: set[int] = set()
    unique_years: list[int] = []
    for yr in entities.years:
        if yr not in seen:
            seen.add(yr)
            unique_years.append(yr)
    entities.years = unique_years

    # -- Court names ---------------------------------------------------------
    lower_raw = raw.lower()
    # Try multi-word courts first (longest match wins)
    matched_courts: list[str] = []
    for canonical in sorted(_COURT_NAMES, key=len, reverse=True):
        if canonical in lower_raw and canonical not in matched_courts:
            matched_courts.append(_COURT_NAMES[canonical])

    # Deduplicate display forms
    seen_display: set[str] = set()
    for display in matched_courts:
        if display not in seen_display:
            entities.courts.append(display)
            seen_display.add(display)

    return entities


# ---------------------------------------------------------------------------
# Stage 3: Intent classification
# ---------------------------------------------------------------------------


def _classify_intent(normalised: str, entities: QueryEntities) -> QueryIntent:
    """Classify the dominant intent of the query."""

    # Citation lookup: has a well-formed citation → high-precision intent
    if entities.citations:
        return QueryIntent.CITATION

    # Judge-specific query
    if entities.judge_names:
        return QueryIntent.JUDGE

    # Year range query with minimal other tokens
    if entities.year_range:
        non_year_tokens = [
            t
            for t in normalised.split()
            if not _YEAR_RE.match(t) and t not in ("between", "and", "to")
        ]
        if len(non_year_tokens) <= 3:
            return QueryIntent.DATE_RANGE

    # Court-only query: contains court name(s) but few other substantive tokens
    if entities.courts:
        court_words = set()
        for c in entities.courts:
            court_words.update(c.lower().split())
        non_court_tokens = [
            t for t in normalised.split() if t not in court_words and t not in _STOP_WORDS
        ]
        # Strip bare years from non_court_tokens
        substantive = [t for t in non_court_tokens if not _YEAR_RE.match(t)]
        if len(substantive) <= 1:
            return QueryIntent.COURT

    # Default: topical full-text search
    tokens = _tokenise(normalised)
    if tokens:
        return QueryIntent.TOPICAL

    return QueryIntent.UNKNOWN


# ---------------------------------------------------------------------------
# Stage 4: Token extraction
# ---------------------------------------------------------------------------


def _tokenise(normalised: str) -> list[str]:
    """Return meaningful tokens from a normalised query string."""
    raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalised)
    return [t for t in raw_tokens if t not in _STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Stage 5: Synonym expansion
# ---------------------------------------------------------------------------


def _build_expansions(tokens: list[str], quoted_phrases: list[str]) -> list[str]:
    """Return candidate expansion terms for the given token list.

    Quoted phrases are excluded from expansion to preserve exact-match intent.
    Expansions exclude any term that already appears in the token set.
    """
    # Freeze exact-phrase words so we don't expand them
    exact_words: set[str] = set()
    for phrase in quoted_phrases:
        exact_words.update(phrase.lower().split())

    token_set = {t.lower() for t in tokens}
    cluster_ids_seen: set[int] = set()
    expansions: list[str] = []

    for token in tokens:
        if token in exact_words:
            continue
        cluster_id = _TERM_TO_CLUSTER.get(token)
        if cluster_id is None or cluster_id in cluster_ids_seen:
            continue
        cluster_ids_seen.add(cluster_id)
        for synonym in _SYNONYM_GROUPS[cluster_id]:
            # Only add single-word synonyms as direct expansions
            if " " in synonym:
                continue
            if synonym.lower() not in token_set:
                expansions.append(synonym)

    return expansions
