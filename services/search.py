"""
Search service for legal cases.

Includes an in-memory LRU cache layer (see ``services.cache``) that
short-circuits the database for repeated queries within a configurable
TTL window.  The cache key is derived from the *full* set of normalised
query parameters so different filters never collide.

Results are re-ranked with Okapi BM25 when a query string is supplied.
The ``relevance`` field in each ``CaseResponse`` reflects the BM25 score
(normalised to [0, 1]).  Without a query, relevance defaults to ``1.0``.
"""

import json
import logging

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Case
from models import CaseDetail, CaseResponse, SearchResponse
from services.cache import SearchCache, search_cache
from services.highlight import highlight_snippet
from services.ranking import BM25Scorer

logger = logging.getLogger("legal_api.search")


async def search_cases(
    db: AsyncSession,
    query: str,
    court: str | None = None,
    year: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: int = 1,
    per_page: int = 20,
    highlight: bool = True,
) -> SearchResponse:
    """
    Search for cases with filters, pagination, and BM25 relevance ranking.

    Results are cached in an LRU cache keyed by the full parameter set.
    Cache hits avoid the database round-trip entirely.

    Relevance scores are computed using Okapi BM25 over the page of
    results returned by the database.  Scores are normalised so the
    highest-scoring document receives ``1.0``; all others are scaled
    proportionally.  When no query is supplied, ``relevance`` is ``1.0``
    for every result.
    """
    # -- Check cache ---------------------------------------------------------
    cache_key = SearchCache.make_key(
        q=query,
        court=court,
        year=year,
        date_from=date_from,
        date_to=date_to,
        page=page,
        per_page=per_page,
        highlight=highlight,
    )

    cached = search_cache.get(cache_key)
    if cached is not None:
        logger.debug("cache hit for key=%s", cache_key[:12])
        return cached

    # -- Cache miss -- query the database ------------------------------------
    logger.debug("cache miss for key=%s, querying DB", cache_key[:12])

    # Build base query
    stmt = select(Case)
    conditions = []

    # Text search (title, headnote, text)
    if query:
        search_term = f"%{query}%"
        conditions.append(
            or_(
                Case.title.ilike(search_term),
                Case.headnote.ilike(search_term),
                Case.text.ilike(search_term),
                Case.citation.ilike(search_term),
            )
        )

    # Filters
    if court:
        conditions.append(Case.court.ilike(f"%{court}%"))

    if year:
        conditions.append(Case.year == year)

    if date_from:
        conditions.append(Case.date >= date_from)

    if date_to:
        conditions.append(Case.date <= date_to)

    # Apply conditions
    if conditions:
        stmt = stmt.where(and_(*conditions))

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = await db.scalar(count_stmt) or 0

    # Pagination
    offset = (page - 1) * per_page
    stmt = stmt.offset(offset).limit(per_page)

    # Execute
    result = await db.execute(stmt)
    cases = result.scalars().all()

    # -- BM25 relevance ranking ----------------------------------------------
    bm25_scores: list[float] = _compute_bm25_scores(query, cases)

    # -- Format results ------------------------------------------------------
    results = []
    for i, case in enumerate(cases):
        # Create snippet from headnote or full text
        source = case.headnote or case.text or ""

        if highlight and query:
            snippet = highlight_snippet(source, query, max_length=300)
        else:
            snippet = source[:300] + "..." if len(source) > 300 else source or None

        results.append(
            CaseResponse(
                id=case.id,
                title=case.title,
                citation=case.citation,
                court=case.court,
                date=case.date,
                snippet=snippet,
                relevance=round(bm25_scores[i], 4),
            )
        )

    # Sort results by BM25 relevance (highest first)
    if query:
        results.sort(key=lambda r: r.relevance or 0.0, reverse=True)

    total_pages = (total + per_page - 1) // per_page

    response = SearchResponse(
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        results=results,
    )

    # -- Store in cache ------------------------------------------------------
    search_cache.put(cache_key, response)

    return response


def _compute_bm25_scores(query: str, cases: list) -> list[float]:
    """Compute normalised BM25 relevance scores for a list of DB case rows.

    Parameters
    ----------
    query:
        The user-supplied search string.
    cases:
        ORM ``Case`` objects returned by the database query.

    Returns
    -------
    list[float]
        Relevance score in ``[0.0, 1.0]`` for each case, in the same order.
        Falls back to ``1.0`` per case when ``query`` is empty.
    """
    n = len(cases)
    if not cases:
        return []

    if not query:
        return [1.0] * n

    # Build corpus: combine all searchable text fields per document
    texts = [
        " ".join(
            filter(
                None,
                [case.title, case.citation, case.headnote, case.text],
            )
        )
        for case in cases
    ]

    scorer = BM25Scorer(texts)
    raw_scores = [scorer.score(query, i) for i in range(n)]

    # Normalise to [0, 1] so the API response is easier to interpret
    max_score = max(raw_scores) if any(s > 0 for s in raw_scores) else 1.0
    if max_score == 0:
        return [0.0] * n
    return [s / max_score for s in raw_scores]


async def get_case_by_id(db: AsyncSession, case_id: str) -> CaseDetail | None:
    """Fetch full case details by ID."""
    stmt = select(Case).where(Case.id == case_id)
    result = await db.execute(stmt)
    case = result.scalar_one_or_none()

    if not case:
        return None

    # Parse judges JSON
    judges = None
    if case.judges:
        try:
            judges = json.loads(case.judges)
        except json.JSONDecodeError:
            judges = [case.judges]

    return CaseDetail(
        id=case.id,
        title=case.title,
        citation=case.citation,
        court=case.court,
        date=case.date,
        judges=judges,
        headnote=case.headnote,
        text=case.text,
    )
