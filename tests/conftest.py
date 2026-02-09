"""
Shared test fixtures for Legal API tests.
"""

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from main import app
from database import get_db


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    """Return a mocked AsyncSession."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture()
def override_db(mock_db_session: AsyncMock):
    """Override the get_db dependency with the mock session."""

    async def _override() -> AsyncGenerator[AsyncSession, None]:
        yield mock_db_session

    app.dependency_overrides[get_db] = _override
    yield mock_db_session
    app.dependency_overrides.clear()


@pytest.fixture()
async def client(override_db) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with mocked DB dependency."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

def make_case_row(
    id: str = "case_001",
    title: str = "Smith v. State",
    citation: str = "2024 SC 445",
    court: str = "Supreme Court",
    date: str = "2024-03-15",
    year: int = 2024,
    judges: str = '["Justice A", "Justice B"]',
    headnote: str = "Brief summary of the case.",
    text: str = "Full judgment text goes here.",
):
    """Build a mock Case-like object (SimpleNamespace-style)."""
    row = MagicMock()
    row.id = id
    row.title = title
    row.citation = citation
    row.court = court
    row.date = date
    row.year = year
    row.judges = judges
    row.headnote = headnote
    row.text = text
    return row
