# tests/test_api.py

import pytest
import io
from httpx import AsyncClient

# Marks all tests in this file as asyncio tests
pytestmark = pytest.mark.asyncio

async def test_health_check(async_client: AsyncClient):
    """Tests the main health check endpoint."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

async def test_file_upload(async_client: AsyncClient):
    """Tests uploading a single file."""
    file_content = "col1,col2\nval1,val2"
    files = {"file": ("test.csv", io.BytesIO(file_content.encode()), "text/csv")}
    
    response = await async_client.post("/api/files/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "test.csv"
    assert data["message"] == "File uploaded and processed successfully."

async def test_list_files(async_client: AsyncClient):
    """Tests listing uploaded files after an upload."""
    # First, upload a file to ensure there's something to list
    file_content = "col1,col2\nval1,val2"
    files = {"file": ("list_test.csv", io.BytesIO(file_content.encode()), "text/csv")}
    await async_client.post("/api/files/upload", files=files)

    # Now, list the files
    response = await async_client.get("/api/files/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["original_filename"] == "list_test.csv"

async def test_start_comprehensive_analysis(async_client: AsyncClient, mock_celery_task):
    """Tests that the comprehensive analysis endpoint correctly dispatches a Celery task."""
    # First, upload a file to get a valid file_id
    file_content = "col1,col2\nval1,val2"
    files = {"file": ("analysis_test.csv", io.BytesIO(file_content.encode()), "text/csv")}
    upload_response = await async_client.post("/api/files/upload", files=files)
    file_id = upload_response.json()["file_id"]

    # Now, start the analysis
    payload = {
        "message": "Analyze my data",
        "session_id": "test_session_123",
        "file_context": [file_id],
        "analysis_mode": "comprehensive"
    }

    response = await async_client.post("/api/v2/analyze/comprehensive", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["task_id"] == "mock_task_id_12345"

    # Verify that our mocked Celery task was called
    mock_celery_task.delay.assert_called_once()

async def test_simple_chat(async_client: AsyncClient, mock_llm):
    """Tests the simple chat endpoint, ensuring it calls the LLM."""
    payload = {
        "message": "Hello, world!",
        "session_id": "test_simple_chat_session",
        "analysis_mode": "simple"
    }

    response = await async_client.post("/api/v2/chat/simple", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "mocked LLM response" in data["response"]

    # Verify that the mocked LLM was called
    mock_llm.assert_awaited_once()
