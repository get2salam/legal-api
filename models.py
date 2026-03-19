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


# ─── Query Understanding Models ───────────────────────────────────────────────


class QueryEntitiesResponse(BaseModel):
    """Structured entities extracted from a query string."""

    citations: list[str]
    courts: list[str]
    years: list[int]
    year_range: tuple[int, int] | None = None
    quoted_phrases: list[str]
    judge_names: list[str]


class QueryAnalysisResponse(BaseModel):
    """Response from the /analyze endpoint.

    Contains the normalised query, detected intent, extracted entities,
    optional synonym expansions, and the final token list.
    """

    original: str
    normalised: str
    intent: str
    entities: QueryEntitiesResponse
    expansions: list[str]
    tokens: list[str]
