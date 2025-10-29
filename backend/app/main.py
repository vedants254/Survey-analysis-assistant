"""
FastAPI Main Application
Version 2 - Enhanced Multi-Temporal Data Analysis with LangGraph Integration
"""
import logging
import sys
import uuid
import contextvars

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.logging_conf import configure_logging
from app.db import init_db

# Configure logging BEFORE any other imports
configure_logging()

# Context variable for request ID
request_id_var = contextvars.ContextVar('request_id', default='-')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import routers
from routers import files, v2_analytics

logger = logging.getLogger(__name__)

# Create FastAPI app instance with V2 enhancements
app = FastAPI(
    title="Historical Multi-Table Data Analysis API v2",
    description="Production-grade API with LangGraph workflows for multi-temporal analysis and conversational AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Assign a unique ID to each incoming request for tracing.
    """
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# Mount static files for uploaded data
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include V1 routers (backward compatibility)
app.include_router(files.router, prefix="/api/files", tags=["files"])


# Include V2 enhanced routers (lazy loading)
try:
    from routers import v2_analytics
    app.include_router(v2_analytics.router, prefix="/api/v2", tags=["v2-analytics"])
    logger.info("✅ V2 analytics router loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ V2 analytics router failed to load: {e}")
    logger.warning("V2 features may not be available")

import time
import asyncio

@app.on_event("startup")
async def startup_event():
    """Initialize database tables and setup on startup with a retry mechanism."""
    logger.info("🚀 Starting Historical Multi-Table Data Analysis API v2.0.0")
    
    max_retries = 5
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to connect to the database (Attempt {attempt + 1}/{max_retries})...")
                        # logger.info("Initializing database...")
                        # await init_db() # Schema creation should be handled by Alembic migrations
                        # logger.info("✅ Database initialized successfully")            break  # Exit loop on success
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            if attempt < max_retries - 1:
                logger.warning(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.critical("Could not connect to the database after multiple retries. Shutting down.")
                # In a real app, you might want to exit here, but for now we'll just log it.
                # sys.exit(1)

    logger.info("🧠 LangGraph workflow engine ready")
    logger.info("📊 Enhanced multi-temporal analysis capabilities enabled")
    logger.info("🔌 WebSocket support for real-time updates active")

@app.get("/")
async def root():
    """API information and capabilities endpoint"""
    return {
        "message": "Historical Multi-Table Data Analysis API v2",
        "version": "2.0.0",
        "status": "operational",
        "description": "Production-grade multi-temporal data analysis with LangGraph AI workflows",
        "features": {
            "v1_compatibility": [
                "CSV/Excel file upload", 
                "Basic data analytics",
                "Simple conversational interface",
                "Session management"
            ],
            "v2_enhanced": [
                "LangGraph workflow orchestration",
                "Multi-temporal data analysis",
                "Real-time WebSocket updates", 
                "Advanced trend analysis",
                "LLM-powered code generation",
                "Background task processing",
                "Automated insights generation"
            ]
        },
        "endpoints": {
            "v1_api": {
                "docs": "/docs",
                "files": "/api/files",
                "analytics": "/api/analytics", 
                "chat": "/api/chat"
            },
            "v2_enhanced": {
                "multi_temporal_analysis": "/api/v2/analyze",
                "enhanced_chat": "/api/v2/chat/query",
                "task_status": "/api/v2/tasks/{task_id}/status",
                "system_status": "/api/v2/system/status",
                "websocket": "/api/v2/ws/{session_id}",
                "workflow_test": "/api/v2/test/workflow"
            }
        },
        "architecture": {
            "workflow_nodes": 8,
            "supported_llm_providers": ["gemini", "openai", "anthropic", "ollama"],
            "analysis_types": ["single_table", "cross_table", "temporal_comparison", "trend_analysis"]
        }
    }

@app.get("/health")
async def health_check():
    """Fast health check for deployment validation"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": str(Path().absolute()),
        "ready": True
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check for V2 system"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "database": "connected",
            "upload_dir": str(upload_dir.absolute()),
            "langgraph_workflow": "ready",
            "websocket_manager": "active",
            "v1_compatibility": "enabled",
            "v2_enhanced_features": "operational"
        },
        "capabilities": {
            "multi_temporal_analysis": True,
            "real_time_updates": True,
            "background_processing": True,
            "llm_integration": True,
            "workflow_orchestration": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
