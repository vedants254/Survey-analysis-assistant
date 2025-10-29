# tests/test_integration.py

import pytest
import uuid
from unittest.mock import patch, AsyncMock

from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from tasks import execute_workflow_task
from models.v2_models import ComprehensiveAnalysis, ChatMessage

# Marks all tests in this file as asyncio tests
pytestmark = pytest.mark.asyncio

async def test_workflow_task_database_interaction(db_session: AsyncSession):
    """
    Tests the Celery task's core logic, ensuring it correctly processes
    a mocked workflow result and saves the final state to the database.
    """
    # 1. Arrange
    task_id = str(uuid.uuid4())
    session_id = "integration_test_session"
    request_id = "trace_id_123"
    
    analysis_request_data = {
        "message": "Test query for integration test",
        "session_id": session_id,
        "file_context": [1, 2],
        "analysis_mode": "comprehensive"
    }

    # Mock the result that the LangGraph workflow would return
    mock_final_workflow_result = {
        "executive_summary": "This is a mock summary.",
        "key_findings": ["Finding 1", "Finding 2"],
        "recommended_actions": ["Action 1"],
        "operation_type": "mock_analysis",
        "confidence_score": 0.95
    }

    # Patch the external dependencies of the task
    with patch('tasks.analysis_workflow.ainvoke', new_callable=AsyncMock) as mock_ainvoke,
         patch('tasks.register_progress_callback') as mock_register,
         patch('tasks.unregister_progress_callback') as mock_unregister:
        
        # Configure the mock to return our predictable result
        mock_ainvoke.return_value = {"final_result": mock_final_workflow_result}

        # 2. Act
        # Execute the task logic directly. .run() executes it in the current process.
        # We pass the arguments that .delay() would normally serialize.
        await execute_workflow_task.run(
            task_id=task_id,
            analysis_request_data=analysis_request_data,
            request_id=request_id
        )

        # 3. Assert
        # Verify that the results were saved to the database
        
        # Check for the main analysis result
        analysis_result_stmt = select(ComprehensiveAnalysis).where(ComprehensiveAnalysis.task_id == task_id)
        analysis_record = (await db_session.execute(analysis_result_stmt)).scalars().first()
        
        assert analysis_record is not None
        assert analysis_record.query == "Test query for integration test"
        assert analysis_record.execution_status == "completed"
        assert analysis_record.analysis_results['executive_summary'] == "This is a mock summary."

        # Check for the assistant's chat message
        message_result_stmt = select(ChatMessage).where(ChatMessage.analysis_results['task_id'] == task_id)
        message_record = (await db_session.execute(message_result_stmt)).scalars().first()

        assert message_record is not None
        assert message_record.message_type == "assistant"
        assert "Comprehensive Analysis Completed" in message_record.content
        assert "This is a mock summary." in message_record.content
