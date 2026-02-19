"""
Pydantic models for request/response schemas.
"""

from pydantic import BaseModel


class CaseBase(BaseModel):
    """Base case fields."""

    id: str
    title: str
    citation: str | None = None
    court: str | None = None
    date: str | None = None


class CaseResponse(CaseBase):
    """Case in search results (abbreviated)."""

    snippet: str | None = None
    relevance: float | None = None


class CaseDetail(CaseBase):
    """Full case details."""

    judges: list[str] | None = None
    headnote: str | None = None
    text: str | None = None
    citations_found: list[str] | None = None

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    """Paginated search results."""

    total: int
    page: int
    per_page: int
    total_pages: int
    results: list[CaseResponse]


class StatsResponse(BaseModel):
    """Overall statistics."""

    total_cases: int
    total_courts: int
    year_range: dict | None = None
    avg_text_length: int | None = None


class CourtStats(BaseModel):
    """Statistics by court."""

    court: str
    count: int


class YearStats(BaseModel):
    """Statistics by year."""

    year: int
    count: int
