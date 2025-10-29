# tests/conftest.py

import asyncio
import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock

# Add the backend directory to the Python path before other imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.main import app
from app.db import get_db
from models.v2_models import Base

# --- Test Database Setup ---

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestAsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency override to use the test database session."""
    async with TestAsyncSessionFactory() as session:
        yield session

# Apply the override to the FastAPI app
app.dependency_overrides[get_db] = override_get_db

@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_database():
    """Set up the test database once per session."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

# --- Test Client Fixture ---

@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a standalone database session for non-API tests."""
    async with TestAsyncSessionFactory() as session:
        yield session


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an async test client for making API requests."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

# --- Mocking Fixtures ---

@pytest.fixture(scope="function")
def mock_celery_task():
    """Mocks the Celery task's .delay() method."""
    mock_task = MagicMock()
    mock_task.delay.return_value = MagicMock(id="mock_task_id_12345")
    
    with patch('backend.routers.v2_analytics.execute_workflow_task', mock_task) as patched_task:
        yield patched_task

@pytest.fixture(scope="function")
def mock_llm():
    """Mocks the LLM manager to avoid real API calls."""
    mock_response = MagicMock()
    mock_response.content = '{"response": "This is a mocked LLM response."}'
    
    with patch('backend.services.llm_providers.llm_manager.generate', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_response
        yield mock_generate
