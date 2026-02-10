"""
Export service — generate CSV and JSONL downloads from search results.
"""

import csv
import io
import json
from typing import Optional

from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database import Case

# Maximum rows for a single export (prevents abuse)
MAX_EXPORT_ROWS = 10_000


async def export_cases_csv(
    db: AsyncSession,
    query: str,
    court: Optional[str] = None,
    year: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = MAX_EXPORT_ROWS,
) -> tuple[str, int]:
    """
    Export matching cases as CSV text.

    Returns (csv_string, row_count).
    """
    cases = await _fetch_export_rows(db, query, court, year, date_from, date_to, limit)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "title", "citation", "court", "date", "year", "judges", "headnote"])

    for c in cases:
        writer.writerow([
            c.id,
            c.title,
            c.citation or "",
            c.court or "",
            c.date or "",
            c.year or "",
            c.judges or "",
            (c.headnote or "")[:500],  # Truncate for CSV readability
        ])

    return buf.getvalue(), len(cases)


async def export_cases_jsonl(
    db: AsyncSession,
    query: str,
    court: Optional[str] = None,
    year: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = MAX_EXPORT_ROWS,
) -> tuple[str, int]:
    """
    Export matching cases as JSONL (one JSON object per line).

    Returns (jsonl_string, row_count).
    """
    cases = await _fetch_export_rows(db, query, court, year, date_from, date_to, limit)

    lines = []
    for c in cases:
        obj = {
            "id": c.id,
            "title": c.title,
            "citation": c.citation,
            "court": c.court,
            "date": c.date,
            "year": c.year,
            "judges": c.judges,
            "headnote": c.headnote,
        }
        lines.append(json.dumps(obj, ensure_ascii=False))

    return "\n".join(lines), len(cases)


# ── Internal ──────────────────────────────────────────────────────────────────


async def _fetch_export_rows(
    db: AsyncSession,
    query: str,
    court: Optional[str],
    year: Optional[int],
    date_from: Optional[str],
    date_to: Optional[str],
    limit: int,
) -> list[Case]:
    """Build query, apply filters, and fetch rows."""
    stmt = select(Case)
    conditions = []

    if query:
        term = f"%{query}%"
        conditions.append(
            or_(
                Case.title.ilike(term),
                Case.headnote.ilike(term),
                Case.text.ilike(term),
                Case.citation.ilike(term),
            )
        )

    if court:
        conditions.append(Case.court.ilike(f"%{court}%"))
    if year:
        conditions.append(Case.year == year)
    if date_from:
        conditions.append(Case.date >= date_from)
    if date_to:
        conditions.append(Case.date <= date_to)

    if conditions:
        stmt = stmt.where(and_(*conditions))

    stmt = stmt.limit(min(limit, MAX_EXPORT_ROWS))

    result = await db.execute(stmt)
    return list(result.scalars().all())
