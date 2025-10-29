"""
V2 Database Models - PostgreSQL Schema for Multi-Temporal Data Analysis
Enhanced for production-grade scaling and advanced analytics capabilities
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float, Boolean, ForeignKey, Index, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any, List
import uuid

Base = declarative_base()

class TimeGranularity(PyEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class FileStatus(PyEnum):
    UPLOADING = "UPLOADING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    ERROR = "ERROR"
    ARCHIVED = "ARCHIVED"

class AnalysisType(PyEnum):
    SINGLE_TABLE = "SINGLE_TABLE"
    CROSS_TABLE = "CROSS_TABLE"
    TEMPORAL_COMPARISON = "TEMPORAL_COMPARISON"
    TREND_ANALYSIS = "TREND_ANALYSIS"

class TaskStatus(PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# ================================
# Core Data Models
# ================================

class DataFile(Base):
    """Enhanced file model for multi-temporal data analysis"""
    __tablename__ = "data_files"
    
    id = Column(Integer, primary_key=True, index=True)
    file_uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # File metadata
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA256 for deduplication
    
    # Data structure information
    columns = Column(JSON, nullable=False)
    row_count = Column(Integer, nullable=False)
    numeric_columns = Column(JSON, default=list)
    categorical_columns = Column(JSON, default=list)
    date_columns = Column(JSON, default=list)
    meta_columns = Column(JSON, default=list)  # JSON columns like Meta field
    
    # Temporal information
    time_period = Column(String(50))  # e.g., "2024-11", "Q4-2024", "2024"
    time_granularity = Column(Enum(TimeGranularity), default=TimeGranularity.MONTHLY)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Processing status
    status = Column(Enum(FileStatus), default=FileStatus.UPLOADING)
    processing_logs = Column(JSON, default=list)
    
    # Metadata
    upload_time = Column(DateTime, server_default=func.now())
    updated_time = Column(DateTime, server_default=func.now(), onupdate=func.now())
    tags = Column(JSON, default=list)
    
    # Relationships
    analysis_results = relationship("AnalysisResult", back_populates="data_file")
    temporal_alignments = relationship("TemporalAlignment", back_populates="data_file")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_data_files_time_period', 'time_period'),
        Index('idx_data_files_granularity', 'time_granularity'),
        Index('idx_data_files_status', 'status'),
        Index('idx_data_files_upload_time', 'upload_time'),
    )

class TemporalAlignment(Base):
    """Track relationships between files for temporal analysis"""
    __tablename__ = "temporal_alignments"
    
    id = Column(Integer, primary_key=True, index=True)
    data_file_id = Column(Integer, ForeignKey("data_files.id"), nullable=False)
    
    # Alignment information
    reference_date = Column(DateTime, nullable=False)
    sequence_order = Column(Integer, nullable=False)  # For ordering files chronologically
    alignment_confidence = Column(Float, default=1.0)  # How confident we are in the alignment
    
    # Common schema mapping
    column_mapping = Column(JSON, default=dict)  # Maps columns across files
    schema_compatibility = Column(JSON, default=dict)  # Compatibility analysis
    
    created_time = Column(DateTime, server_default=func.now())
    
    # Relationships
    data_file = relationship("DataFile", back_populates="temporal_alignments")
    
    __table_args__ = (
        Index('idx_temporal_alignments_file_id', 'data_file_id'),
        Index('idx_temporal_alignments_date', 'reference_date'),
    )

# ================================
# Analysis and Results
# ================================

class AnalysisSession(Base):
    """Enhanced session management for multi-turn conversations"""
    __tablename__ = "analysis_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Session metadata
    created_at = Column(DateTime, server_default=func.now())
    last_activity = Column(DateTime, server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Context tracking
    context_files = Column(JSON, default=list)  # List of file IDs in context
    conversation_history = Column(JSON, default=list)  # Simplified history
    session_metadata = Column(JSON, default=dict)
    
    # Performance tracking
    total_queries = Column(Integer, default=0)
    successful_queries = Column(Integer, default=0)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session")
    analysis_results = relationship("AnalysisResult", back_populates="session")
    background_tasks = relationship("BackgroundTask", back_populates="session")
    
    __table_args__ = (
        Index('idx_analysis_sessions_uuid', 'session_uuid'),
        Index('idx_analysis_sessions_activity', 'last_activity'),
    )

class ChatMessage(Base):
    """Enhanced message model with LangGraph integration"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("analysis_sessions.id"), nullable=False)
    
    # Message content
    message_type = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    # LangGraph tracking
    graph_execution_id = Column(String(100), nullable=True)  # LangGraph execution ID
    graph_state = Column(JSON, nullable=True)  # Current graph state
    node_outputs = Column(JSON, nullable=True)  # Outputs from each node
    
    # Analysis context
    analysis_intent = Column(JSON, nullable=True)  # Parsed intent
    related_files = Column(JSON, default=list)  # Files referenced in this message
    analysis_results = Column(JSON, nullable=True)  # Inline results
    
    # Timing and performance
    timestamp = Column(DateTime, server_default=func.now())
    processing_time = Column(Float, nullable=True)  # Time taken to process
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="messages")
    
    __table_args__ = (
        Index('idx_chat_messages_session', 'session_id'),
        Index('idx_chat_messages_timestamp', 'timestamp'),
        Index('idx_chat_messages_graph_execution', 'graph_execution_id'),
    )

class AnalysisResult(Base):
    """Comprehensive analysis results with caching"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    result_uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Link to context
    session_id = Column(Integer, ForeignKey("analysis_sessions.id"), nullable=False)
    data_file_id = Column(Integer, ForeignKey("data_files.id"), nullable=True)  # Primary file
    
    # Analysis metadata
    analysis_type = Column(Enum(AnalysisType), nullable=False)
    operation_name = Column(String(100), nullable=False)
    parameters = Column(JSON, default=dict)
    
    # Multi-file context
    involved_files = Column(JSON, default=list)  # All files involved in analysis
    time_range = Column(JSON, nullable=True)  # Start and end times
    
    # Results
    result_data = Column(JSON, nullable=False)  # Primary results
    derived_insights = Column(JSON, default=list)  # LLM-generated insights
    recommended_actions = Column(JSON, default=list)  # Suggested next steps
    confidence_score = Column(Float, default=0.0)  # Result confidence
    
    # Code generation
    generated_code = Column(Text, nullable=True)  # Code used for analysis
    execution_log = Column(JSON, default=list)  # Execution details
    
    # Performance and caching
    created_at = Column(DateTime, server_default=func.now())
    computation_time = Column(Float, nullable=False)  # Time taken to compute
    cache_key = Column(String(200), nullable=True, index=True)  # For result caching
    is_cached = Column(Boolean, default=False)
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="analysis_results")
    data_file = relationship("DataFile", back_populates="analysis_results")
    
    __table_args__ = (
        Index('idx_analysis_results_session', 'session_id'),
        Index('idx_analysis_results_type', 'analysis_type'),
        Index('idx_analysis_results_cache', 'cache_key'),
        Index('idx_analysis_results_created', 'created_at'),
    )

# ================================
# Background Processing
# ================================

class BackgroundTask(Base):
    """Background task tracking for async operations"""
    __tablename__ = "background_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    session_id = Column(Integer, ForeignKey("analysis_sessions.id"), nullable=True)
    
    # Task metadata
    task_type = Column(String(50), nullable=False)  # file_processing, analysis, etc.
    task_name = Column(String(100), nullable=False)
    parameters = Column(JSON, default=dict)
    
    # Status tracking
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    status_message = Column(String(500), nullable=True)
    
    # Results
    result_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    logs = Column(JSON, default=list)
    
    # Timing
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Celery integration
    celery_task_id = Column(String(100), nullable=True, unique=True)
    
    # Relationships
    session = relationship("AnalysisSession", back_populates="background_tasks")
    
    __table_args__ = (
        Index('idx_background_tasks_status', 'status'),
        Index('idx_background_tasks_type', 'task_type'),
        Index('idx_background_tasks_celery', 'celery_task_id'),
    )

# ================================
# LangGraph State Management
# ================================

class GraphExecution(Base):
    """Track LangGraph workflow executions"""
    __tablename__ = "graph_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    execution_uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    session_id = Column(Integer, ForeignKey("analysis_sessions.id"), nullable=False)
    
    # Graph metadata
    graph_name = Column(String(100), nullable=False)
    input_query = Column(Text, nullable=False)
    
    # Execution tracking
    current_node = Column(String(50), nullable=True)
    completed_nodes = Column(JSON, default=list)
    node_outputs = Column(JSON, default=dict)
    graph_state = Column(JSON, default=dict)
    
    # Results
    final_result = Column(JSON, nullable=True)
    execution_path = Column(JSON, default=list)  # Path taken through graph
    
    # Status and timing
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    total_time = Column(Float, nullable=True)
    
    # Error handling
    error_node = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_graph_executions_session', 'session_id'),
        Index('idx_graph_executions_status', 'status'),
        Index('idx_graph_executions_started', 'started_at'),
    )

# ================================
# Caching and Performance
# ================================

class QueryCache(Base):
    """Cache for frequently accessed analysis results"""
    __tablename__ = "query_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(200), unique=True, nullable=False)
    
    # Query metadata
    query_hash = Column(String(64), nullable=False)  # SHA256 of normalized query
    parameters_hash = Column(String(64), nullable=False)  # SHA256 of parameters
    
    # Cache data
    result_data = Column(JSON, nullable=False)
    result_metadata = Column(JSON, default=dict)
    
    # Cache management
    created_at = Column(DateTime, server_default=func.now())
    last_accessed = Column(DateTime, server_default=func.now())
    access_count = Column(Integer, default=0)
    ttl = Column(Integer, default=3600)  # TTL in seconds
    
    __table_args__ = (
        Index('idx_query_cache_key', 'cache_key'),
        Index('idx_query_cache_hash', 'query_hash'),
        Index('idx_query_cache_accessed', 'last_accessed'),
    )

# ================================
# System Monitoring
# ================================

class SystemMetrics(Base):
    """System performance and usage metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metric metadata
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    
    # Metric data
    value = Column(Float, nullable=False)
    labels = Column(JSON, default=dict)  # Labels for metric
    
    # Timing
    timestamp = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_system_metrics_name', 'metric_name'),
        Index('idx_system_metrics_timestamp', 'timestamp'),
    )