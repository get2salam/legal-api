"""
Tests for the /api/v1/analyze endpoint.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_analyze_topical_query(client: AsyncClient):
    """GET /api/v1/analyze returns analysis for a topical query."""
    resp = await client.get("/api/v1/analyze", params={"q": "contract breach damages 2021"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["original"] == "contract breach damages 2021"
    assert "normalised" in data
    assert data["intent"] in ("topical", "citation", "judge", "court", "date_range", "unknown")
    assert isinstance(data["tokens"], list)
    assert isinstance(data["expansions"], list)
    assert "entities" in data
    entities = data["entities"]
    assert "years" in entities
    assert 2021 in entities["years"]


@pytest.mark.anyio
async def test_analyze_citation_query(client: AsyncClient):
    """Citation query returns intent=citation and populated citations list."""
    resp = await client.get("/api/v1/analyze", params={"q": "2021 SC 45"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "citation"
    assert "2021 SC 45" in data["entities"]["citations"]


@pytest.mark.anyio
async def test_analyze_missing_q(client: AsyncClient):
    """GET /api/v1/analyze without q returns 422."""
    resp = await client.get("/api/v1/analyze")
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_analyze_court_entities(client: AsyncClient):
    """Court names are extracted into entities.courts."""
    resp = await client.get("/api/v1/analyze", params={"q": "Supreme Court 2022"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Supreme Court" in data["entities"]["courts"]


@pytest.mark.anyio
async def test_analyze_year_range(client: AsyncClient):
    """Year range queries return date_range intent and year_range entity."""
    resp = await client.get("/api/v1/analyze", params={"q": "between 2018 and 2022"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "date_range"
    assert data["entities"]["year_range"] == [2018, 2022]


@pytest.mark.anyio
async def test_analyze_quoted_phrase(client: AsyncClient):
    """Quoted phrases are extracted correctly."""
    resp = await client.get("/api/v1/analyze", params={"q": '"breach of trust" Supreme Court'})
    assert resp.status_code == 200
    data = resp.json()
    assert "breach of trust" in data["entities"]["quoted_phrases"]


@pytest.mark.anyio
async def test_analyze_expansion_present(client: AsyncClient):
    """Topical queries return synonym expansions."""
    resp = await client.get("/api/v1/analyze", params={"q": "negligence damages"})
    assert resp.status_code == 200
    data = resp.json()
    # Should have at least one expansion for known legal terms
    assert isinstance(data["expansions"], list)
