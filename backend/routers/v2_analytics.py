
"""
V2 Unified Analytics Router
Consolidates all API logic for files, chat, and analysis into a single, modern router.
"""

import asyncio
import uuid
import json
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db import get_db
from models.v2_models import DataFile, AnalysisSession, ChatMessage
from services.llm_providers import llm_manager, LLMMessage
from tasks import execute_workflow_task

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Models ---

class AnalysisRequest(BaseModel):
    file_id: int
    operation: str
    column: str
    group_by: Optional[str] = None

class EnhancedChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str
    file_context: Optional[List[int]] = None
    analysis_mode: str = Field(default="simple", description="simple or comprehensive")

# --- WebSocket Connection Manager ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_json(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(data)

manager = ConnectionManager()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)

# --- Endpoints ---

@router.post("/analyze/comprehensive")
async def comprehensive_analysis(request: EnhancedChatRequest, db: AsyncSession = Depends(get_db)):
    if not request.file_context:
        raise HTTPException(status_code=400, detail="Comprehensive analysis requires file context.")

    # Ensure session exists
    session = await db.execute(select(AnalysisSession).where(AnalysisSession.session_uuid == request.session_id))
    if not session.scalars().first():
        db.add(AnalysisSession(session_uuid=request.session_id))
        await db.commit()

    # Save user message
    user_message = ChatMessage(session_id=request.session_id, message_type="user", content=request.message)
    db.add(user_message)
    await db.commit()

    # Dispatch Celery task
    task = execute_workflow_task.delay(
        task_id=str(uuid.uuid4()),
        analysis_request_data=request.dict(),
        request_id=str(uuid.uuid4()) # This should come from middleware
    )

    return {
        "message": "Comprehensive analysis workflow initiated.",
        "task_id": task.id,
    }

@router.post("/chat/simple")
async def simple_chat(request: EnhancedChatRequest, db: AsyncSession = Depends(get_db)):
    # Ensure session exists
    session_res = await db.execute(select(AnalysisSession).where(AnalysisSession.session_uuid == request.session_id))
    session = session_res.scalars().first()
    if not session:
        session = AnalysisSession(session_uuid=request.session_id)
        db.add(session)
        await db.commit()
        await db.refresh(session)

    # Save user message
    user_message = ChatMessage(session_id=session.id, message_type="user", content=request.message)
    db.add(user_message)
    await db.commit()

    # Generate simple response
    # This is a simplified version of the logic from the old chat.py
    system_prompt = "You are a helpful data analysis assistant. Be concise."
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=request.message)
    ]
    llm_response = await llm_manager.generate(messages)
    
    # Save assistant message
    assistant_message = ChatMessage(session_id=session.id, message_type="assistant", content=llm_response.content)
    db.add(assistant_message)
    await db.commit()

    return {"response": llm_response.content, "session_id": request.session_id}


@router.get("/chat/examples")
async def get_example_queries():
    return {
        "categories": {
            "Basic Analytics": ["What's the average revenue?", "Show me total sales"],
            "Grouped Analysis": ["Average revenue by region", "Total units by product"],
            "Comparisons": ["Compare revenue and units", "Show revenue vs discount correlation"],
            "Data Exploration": ["Analyze my data", "Show me a summary"],
        }
    }

@router.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Check the status of a Celery task."""
    from celery.result import AsyncResult
    from celery_app import celery_app
    
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.successful():
        response["result"] = task_result.get()
    elif task_result.failed():
        # In a real app, you'd want to sanitize this error
        response["result"] = str(task_result.info)
        
    return response
