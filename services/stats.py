"""
Statistics service for legal cases.
"""

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import Case
from models import StatsResponse


async def get_statistics(db: AsyncSession) -> StatsResponse:
    """Get overall statistics."""
    # Total cases
    total = await db.scalar(select(func.count(Case.id))) or 0
    
    # Unique courts
    courts = await db.scalar(select(func.count(func.distinct(Case.court)))) or 0
    
    # Year range
    min_year = await db.scalar(select(func.min(Case.year)))
    max_year = await db.scalar(select(func.max(Case.year)))
    
    year_range = None
    if min_year and max_year:
        year_range = {"min": min_year, "max": max_year}
    
    # Average text length
    avg_length = await db.scalar(select(func.avg(func.length(Case.text))))
    
    return StatsResponse(
        total_cases=total,
        total_courts=courts,
        year_range=year_range,
        avg_text_length=int(avg_length) if avg_length else None,
    )


async def get_court_stats(db: AsyncSession) -> list[dict]:
    """Get case count by court."""
    stmt = (
        select(Case.court, func.count(Case.id).label("count"))
        .group_by(Case.court)
        .order_by(func.count(Case.id).desc())
    )
    
    result = await db.execute(stmt)
    rows = result.all()
    
    return [{"court": row[0] or "Unknown", "count": row[1]} for row in rows]


async def get_year_stats(db: AsyncSession) -> list[dict]:
    """Get case count by year."""
    stmt = (
        select(Case.year, func.count(Case.id).label("count"))
        .where(Case.year.isnot(None))
        .group_by(Case.year)
        .order_by(Case.year.desc())
    )
    
    result = await db.execute(stmt)
    rows = result.all()
    
    return [{"year": row[0], "count": row[1]} for row in rows]
