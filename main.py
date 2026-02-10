"""
Legal API - FastAPI REST backend for case law search.

A generic, jurisdiction-agnostic API for legal research.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from database import init_db, get_db, AsyncSession
from models import CaseResponse, CaseDetail, SearchResponse, StatsResponse
from services.search import search_cases, get_case_by_id
from services.stats import get_statistics, get_court_stats, get_year_stats
from services.export import export_cases_csv, export_cases_jsonl
from middleware import RequestLoggingMiddleware, setup_logging

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""
    api_title: str = "Legal Case Law API"
    api_version: str = "1.0.0"
    database_url: str = "sqlite+aiosqlite:///./legal.db"
    per_page_default: int = 20
    per_page_max: int = 100
    api_key_enabled: bool = False
    api_key: str = ""
    cors_origins: list[str] = ["*"]

    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()

# Structured request logging
setup_logging(settings.log_level)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# API Key authentication (optional)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if authentication is enabled."""
    if not settings.api_key_enabled:
        return True
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db(settings.database_url)
    yield


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="REST API for legal case law search and retrieval",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request logging with correlation IDs
app.add_middleware(RequestLoggingMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Search Endpoints ────────────────────────────────────────────────────────


@app.get("/api/v1/search", response_model=SearchResponse)
@limiter.limit("60/minute")
async def search(
    request: Request,
    q: str = Query(..., description="Search query"),
    court: Optional[str] = Query(None, description="Filter by court"),
    year: Optional[int] = Query(None, description="Filter by year"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(None, ge=1, le=100, description="Results per page"),
    highlight: bool = Query(True, description="Highlight matching terms in snippets"),
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """
    Search for legal cases.
    
    Returns paginated results with relevance ranking.
    Set ``highlight=false`` to disable ``<mark>`` tag wrapping in snippets.
    """
    per_page = per_page or settings.per_page_default
    per_page = min(per_page, settings.per_page_max)
    
    results = await search_cases(
        db=db,
        query=q,
        court=court,
        year=year,
        date_from=date_from,
        date_to=date_to,
        page=page,
        per_page=per_page,
        highlight=highlight,
    )
    
    return results


# ─── Case Endpoints ──────────────────────────────────────────────────────────


@app.get("/api/v1/cases/{case_id}", response_model=CaseDetail)
async def get_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """
    Get full case details by ID.
    """
    case = await get_case_by_id(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


# ─── Statistics Endpoints ────────────────────────────────────────────────────


@app.get("/api/v1/stats", response_model=StatsResponse)
async def stats(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """
    Get overall statistics.
    """
    return await get_statistics(db)


@app.get("/api/v1/stats/courts")
async def stats_by_court(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """
    Get case count by court.
    """
    return await get_court_stats(db)


@app.get("/api/v1/stats/years")
async def stats_by_year(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """
    Get case count by year.
    """
    return await get_year_stats(db)


# ─── Export Endpoints ─────────────────────────────────────────────────────────


@app.get("/api/v1/export/csv")
@limiter.limit("10/minute")
async def export_csv(
    request: Request,
    q: str = Query(..., description="Search query"),
    court: Optional[str] = Query(None, description="Filter by court"),
    year: Optional[int] = Query(None, description="Filter by year"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, ge=1, le=10000, description="Max rows"),
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """
    Export search results as CSV.

    Returns a downloadable CSV file with matching cases.
    """
    csv_text, count = await export_cases_csv(
        db=db, query=q, court=court, year=year,
        date_from=date_from, date_to=date_to, limit=limit,
    )
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=legal-export-{count}-cases.csv",
            "X-Export-Count": str(count),
        },
    )


@app.get("/api/v1/export/jsonl")
@limiter.limit("10/minute")
async def export_jsonl(
    request: Request,
    q: str = Query(..., description="Search query"),
    court: Optional[str] = Query(None, description="Filter by court"),
    year: Optional[int] = Query(None, description="Filter by year"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, ge=1, le=10000, description="Max rows"),
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """
    Export search results as JSONL (one JSON object per line).

    Useful for bulk ingestion into data pipelines or vector databases.
    """
    jsonl_text, count = await export_cases_jsonl(
        db=db, query=q, court=court, year=year,
        date_from=date_from, date_to=date_to, limit=limit,
    )
    return PlainTextResponse(
        content=jsonl_text,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f"attachment; filename=legal-export-{count}-cases.jsonl",
            "X-Export-Count": str(count),
        },
    )


# ─── Utility Endpoints ───────────────────────────────────────────────────────


@app.get("/api/v1/courts")
async def list_courts(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """
    List all available courts.
    """
    courts = await get_court_stats(db)
    return {"courts": [c["court"] for c in courts]}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
