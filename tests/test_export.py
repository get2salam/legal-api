"""
Tests for export and highlight features.
"""

import pytest
from httpx import AsyncClient

from services.highlight import highlight_snippet


# ─── CSV Export ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_csv(client: AsyncClient):
    """CSV export returns valid CSV with headers."""
    resp = await client.get("/api/v1/export/csv", params={"q": "contract"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    lines = resp.text.strip().split("\n")
    assert lines[0] == "id,title,citation,court,date,year,judges,headnote"
    assert len(lines) >= 2  # header + at least one data row


@pytest.mark.asyncio
async def test_export_csv_count_header(client: AsyncClient):
    """CSV export includes X-Export-Count header."""
    resp = await client.get("/api/v1/export/csv", params={"q": "text"})
    assert "X-Export-Count" in resp.headers
    assert int(resp.headers["X-Export-Count"]) >= 1


@pytest.mark.asyncio
async def test_export_csv_with_filters(client: AsyncClient):
    """CSV export respects court filter."""
    resp = await client.get(
        "/api/v1/export/csv", params={"q": "text", "court": "Supreme Court"}
    )
    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    for line in lines[1:]:
        assert "Supreme Court" in line


@pytest.mark.asyncio
async def test_export_csv_no_results(client: AsyncClient):
    """CSV export with no matches returns header only."""
    resp = await client.get("/api/v1/export/csv", params={"q": "xyznonexistent"})
    assert resp.status_code == 200
    lines = resp.text.strip().split("\n")
    assert len(lines) == 1  # header only


# ─── JSONL Export ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_jsonl(client: AsyncClient):
    """JSONL export returns valid JSON lines."""
    import json

    resp = await client.get("/api/v1/export/jsonl", params={"q": "contract"})
    assert resp.status_code == 200
    assert "ndjson" in resp.headers["content-type"]
    lines = [l for l in resp.text.strip().split("\n") if l]
    assert len(lines) >= 1
    obj = json.loads(lines[0])
    assert "id" in obj
    assert "title" in obj


# ─── Search Highlighting ─────────────────────────────────────────────────────


def test_highlight_single_term():
    """Single term is wrapped in <mark> tags."""
    result = highlight_snippet("The contract was breached.", "contract", max_length=200)
    assert "<mark>contract</mark>" in result


def test_highlight_multi_term():
    """Multiple query tokens are each highlighted."""
    result = highlight_snippet(
        "The contract dispute involved a breach of agreement.",
        "contract breach",
        max_length=200,
    )
    assert "<mark>contract</mark>" in result
    assert "<mark>breach</mark>" in result


def test_highlight_case_insensitive():
    """Highlighting is case-insensitive."""
    result = highlight_snippet("CONTRACT law is complex.", "contract", max_length=200)
    assert "<mark>CONTRACT</mark>" in result


def test_highlight_no_match():
    """When query doesn't match, original text is returned truncated."""
    result = highlight_snippet("Some unrelated text here.", "quantum", max_length=200)
    assert "<mark>" not in result
    assert "Some unrelated text" in result


def test_highlight_empty_text():
    """Empty text returns None."""
    result = highlight_snippet(None, "test")
    assert result is None


def test_highlight_empty_query():
    """Empty query returns truncated text without marks."""
    result = highlight_snippet("Some text content.", "", max_length=10)
    assert "<mark>" not in result


def test_highlight_long_text_centering():
    """Snippet is centred around the first match in long text."""
    text = "A" * 500 + " contract " + "B" * 500
    result = highlight_snippet(text, "contract", max_length=100)
    assert "<mark>contract</mark>" in result
    assert result.startswith("...")  # Trimmed from start


@pytest.mark.asyncio
async def test_search_with_highlight(client: AsyncClient):
    """Search endpoint returns highlighted snippets by default."""
    resp = await client.get("/api/v1/search", params={"q": "contract"})
    data = resp.json()
    if data["results"]:
        # At least one result should have <mark> in snippet
        snippets = [r["snippet"] for r in data["results"] if r.get("snippet")]
        assert any("<mark>" in s for s in snippets)


@pytest.mark.asyncio
async def test_search_no_highlight(client: AsyncClient):
    """Search with highlight=false returns plain snippets."""
    resp = await client.get(
        "/api/v1/search", params={"q": "contract", "highlight": "false"}
    )
    data = resp.json()
    for r in data["results"]:
        if r.get("snippet"):
            assert "<mark>" not in r["snippet"]


# ─── Middleware ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_id_header(client: AsyncClient):
    """Response includes X-Request-ID header."""
    resp = await client.get("/health")
    assert "X-Request-ID" in resp.headers


@pytest.mark.asyncio
async def test_custom_request_id_preserved(client: AsyncClient):
    """Client-provided X-Request-ID is echoed back."""
    resp = await client.get("/health", headers={"X-Request-ID": "my-custom-id-123"})
    assert resp.headers["X-Request-ID"] == "my-custom-id-123"


@pytest.mark.asyncio
async def test_response_time_header(client: AsyncClient):
    """Response includes X-Response-Time header."""
    resp = await client.get("/health")
    assert "X-Response-Time" in resp.headers
    assert resp.headers["X-Response-Time"].endswith("ms")
