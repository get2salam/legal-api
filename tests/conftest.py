"""
Shared test fixtures for Legal API tests.
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def make_case_row(
    id: str = "case_001",
    title: str = "Smith v. State",
    citation: str = "2024 SC 445",
    court: str = "Supreme Court",
    date: str = "2024-03-15",
    year: int = 2024,
    judges: str = '["Justice A", "Justice B"]',
    headnote: str = "Brief summary of the contract case.",
    text: str = "Full judgment text about the contract dispute.",
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


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    """Return a mocked AsyncSession with a synchronous execute() result.

    Python 3.14 changed AsyncMock so that child attributes of an AsyncMock are
    also AsyncMock instances.  This means that the return value of
    ``await session.execute(stmt)`` would itself be an AsyncMock, making
    ``result.scalars()`` return a coroutine instead of a sync ScalarResult.

    To stay compatible across Python versions we explicitly wire up the
    execute() return value as a plain MagicMock with realistic sub-attributes.
    """
    session = AsyncMock(spec=AsyncSession)

    # Build a synchronous result object that mimics AsyncResult / CursorResult.
    mock_result = MagicMock()
    _default_rows = [make_case_row()]
    mock_result.scalars.return_value.all.return_value = _default_rows
    # scalar() is used for COUNT queries -- return 1 matching row by default.
    mock_result.scalar.return_value = 1

    # Override the AsyncMock return value so awaiting execute() gives a sync obj.
    session.execute.return_value = mock_result

    # scalar() on the session itself (used in search_cases COUNT query).
    session.scalar.return_value = 1

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
