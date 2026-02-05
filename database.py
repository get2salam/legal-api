"""
Database setup and connection management.
"""

from typing import AsyncGenerator
from sqlalchemy import Column, String, Integer, Text, Date, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Global engine and session maker
engine = None
async_session = None


class Case(Base):
    """Case law database model."""
    __tablename__ = "cases"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    citation = Column(String, index=True)
    court = Column(String, index=True)
    date = Column(String, index=True)  # ISO format
    year = Column(Integer, index=True)
    judges = Column(Text)  # JSON array
    headnote = Column(Text)
    text = Column(Text)


async def init_db(database_url: str):
    """Initialize database connection."""
    global engine, async_session
    
    # Convert sqlite URL for async
    if database_url.startswith("sqlite:///"):
        database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    
    engine = create_async_engine(database_url, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session() as session:
        yield session
