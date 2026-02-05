"""
Search service for legal cases.
"""

import json
from typing import Optional
from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database import Case
from models import SearchResponse, CaseResponse, CaseDetail


async def search_cases(
    db: AsyncSession,
    query: str,
    court: Optional[str] = None,
    year: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
) -> SearchResponse:
    """
    Search for cases with filters and pagination.
    
    Uses simple LIKE matching. For production, consider:
    - Full-text search (PostgreSQL tsvector, SQLite FTS5)
    - Elasticsearch integration
    - Semantic search with embeddings
    """
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
        # Create snippet from headnote or text
        snippet = case.headnote or (case.text[:300] + "..." if case.text else None)
        
        results.append(CaseResponse(
            id=case.id,
            title=case.title,
            citation=case.citation,
            court=case.court,
            date=case.date,
            snippet=snippet,
            relevance=1.0,  # Simple ranking; enhance with BM25/TF-IDF
        ))
    
    total_pages = (total + per_page - 1) // per_page
    
    return SearchResponse(
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        results=results,
    )


async def get_case_by_id(db: AsyncSession, case_id: str) -> Optional[CaseDetail]:
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
