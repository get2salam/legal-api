"""
Search service for legal cases.

Includes an in-memory LRU cache layer (see ``services.cache``) that
short-circuits the database for repeated queries within a configurable
TTL window.  The cache key is derived from the *full* set of normalised
query parameters so different filters never collide.
"""

import json
import logging

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Case
from models import CaseDetail, CaseResponse, SearchResponse
from services.cache import SearchCache, search_cache
from services.highlight import highlight_snippet

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
    Search for cases with filters and pagination.

    Results are cached in an LRU cache keyed by the full parameter set.
    Cache hits avoid the database round-trip entirely.

    Uses simple LIKE matching. For production, consider:
    - Full-text search (PostgreSQL tsvector, SQLite FTS5)
    - Elasticsearch integration
    - Semantic search with embeddings
    """
    # ── Check cache ──────────────────────────────────────────────────────
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

    # ── Cache miss — query the database ──────────────────────────────────
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

    # Format results
    results = []
    for case in cases:
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
                relevance=1.0,  # Simple ranking; enhance with BM25/TF-IDF
            )
        )

    total_pages = (total + per_page - 1) // per_page

    response = SearchResponse(
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        results=results,
    )

    # ── Store in cache ───────────────────────────────────────────────────
    search_cache.put(cache_key, response)

    return response


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
