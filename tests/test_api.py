"""
Unit tests for Legal API endpoints.

All database calls are mocked — no real DB is required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from tests.conftest import make_case_row

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health(client: AsyncClient):
    """GET /health returns 200 with status healthy."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@patch("main.search_cases")
async def test_search_returns_results(mock_search, client: AsyncClient):
    """GET /api/v1/search?q=... returns paginated results."""
    mock_search.return_value = {
        "total": 1,
        "page": 1,
        "per_page": 20,
        "total_pages": 1,
        "results": [
            {
                "id": "case_001",
                "title": "Smith v. State",
                "citation": "2024 SC 445",
                "court": "Supreme Court",
                "date": "2024-03-15",
                "snippet": "Brief summary of the case.",
                "relevance": 1.0,
            }
        ],
    }

    resp = await client.get("/api/v1/search", params={"q": "constitutional"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == "case_001"


@pytest.mark.anyio
async def test_search_missing_query(client: AsyncClient):
    """GET /api/v1/search without q returns 422."""
    resp = await client.get("/api/v1/search")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Get Case
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@patch("main.get_case_by_id")
async def test_get_case_found(mock_get, client: AsyncClient):
    """GET /api/v1/cases/{id} returns case details when found."""
    mock_get.return_value = {
        "id": "case_001",
        "title": "Smith v. State",
        "citation": "2024 SC 445",
        "court": "Supreme Court",
        "date": "2024-03-15",
        "judges": ["Justice A", "Justice B"],
        "headnote": "Brief summary of the case.",
        "text": "Full judgment text goes here.",
        "citations_found": None,
    }

    resp = await client.get("/api/v1/cases/case_001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "case_001"
    assert data["title"] == "Smith v. State"


@pytest.mark.anyio
@patch("main.get_case_by_id")
async def test_get_case_not_found(mock_get, client: AsyncClient):
    """GET /api/v1/cases/{id} returns 404 for missing case."""
    mock_get.return_value = None

    resp = await client.get("/api/v1/cases/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@patch("main.get_statistics")
async def test_stats(mock_stats, client: AsyncClient):
    """GET /api/v1/stats returns statistics."""
    mock_stats.return_value = {
        "total_cases": 500,
        "total_courts": 12,
        "year_range": {"min": 1950, "max": 2024},
        "avg_text_length": 4200,
    }

    resp = await client.get("/api/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_cases"] == 500
    assert data["total_courts"] == 12


@pytest.mark.anyio
@patch("main.get_court_stats")
async def test_stats_courts(mock_court, client: AsyncClient):
    """GET /api/v1/stats/courts returns court breakdown."""
    mock_court.return_value = [
        {"court": "Supreme Court", "count": 200},
        {"court": "High Court", "count": 300},
    ]

    resp = await client.get("/api/v1/stats/courts")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2


@pytest.mark.anyio
@patch("main.get_year_stats")
async def test_stats_years(mock_year, client: AsyncClient):
    """GET /api/v1/stats/years returns year breakdown."""
    mock_year.return_value = [
        {"year": 2024, "count": 50},
        {"year": 2023, "count": 80},
    ]

    resp = await client.get("/api/v1/stats/years")
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["year"] == 2024


# ---------------------------------------------------------------------------
# Courts list
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@patch("main.get_court_stats")
async def test_list_courts(mock_court, client: AsyncClient):
    """GET /api/v1/courts returns list of court names."""
    mock_court.return_value = [
        {"court": "Supreme Court", "count": 200},
        {"court": "High Court", "count": 300},
    ]

    resp = await client.get("/api/v1/courts")
    assert resp.status_code == 200
    data = resp.json()
    assert "courts" in data
    assert "Supreme Court" in data["courts"]
