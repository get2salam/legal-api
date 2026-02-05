"""
Pydantic models for request/response schemas.
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel


class CaseBase(BaseModel):
    """Base case fields."""
    id: str
    title: str
    citation: Optional[str] = None
    court: Optional[str] = None
    date: Optional[str] = None


class CaseResponse(CaseBase):
    """Case in search results (abbreviated)."""
    snippet: Optional[str] = None
    relevance: Optional[float] = None


class CaseDetail(CaseBase):
    """Full case details."""
    judges: Optional[list[str]] = None
    headnote: Optional[str] = None
    text: Optional[str] = None
    citations_found: Optional[list[str]] = None
    
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
    year_range: Optional[dict] = None
    avg_text_length: Optional[int] = None


class CourtStats(BaseModel):
    """Statistics by court."""
    court: str
    count: int


class YearStats(BaseModel):
    """Statistics by year."""
    year: int
    count: int
