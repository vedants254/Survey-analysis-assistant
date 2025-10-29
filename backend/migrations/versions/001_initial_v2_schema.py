"""Initial V2 schema - multi-temporal data analysis

Revision ID: 001_initial_v2
Revises: 
Create Date: 2024-09-13 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial_v2'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create all V2 tables for multi-temporal data analysis"""
    
    # Create ENUM types
    op.execute("CREATE TYPE timegranularity AS ENUM ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')")
    op.execute("CREATE TYPE filestatus AS ENUM ('uploading', 'processing', 'ready', 'error', 'archived')")
    op.execute("CREATE TYPE analysistype AS ENUM ('single_table', 'cross_table', 'temporal_comparison', 'trend_analysis')")
    op.execute("CREATE TYPE taskstatus AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled')")
    
    # Create data_files table
    op.create_table('data_files',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('file_uuid', sa.String(length=36), nullable=False, unique=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('columns', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=False),
        sa.Column('numeric_columns', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('categorical_columns', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('date_columns', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('meta_columns', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('time_period', sa.String(length=50)),
        sa.Column('time_granularity', sa.Enum(name='timegranularity'), default='monthly'),
        sa.Column('start_date', sa.DateTime(timezone=True)),
        sa.Column('end_date', sa.DateTime(timezone=True)),
        sa.Column('status', sa.Enum(name='filestatus'), default='uploading'),
        sa.Column('processing_logs', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('upload_time', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_time', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('tags', postgresql.JSON(astext_type=sa.Text())),
    )
    
    # Create indexes for data_files
    op.create_index('idx_data_files_time_period', 'data_files', ['time_period'])
    op.create_index('idx_data_files_granularity', 'data_files', ['time_granularity'])
    op.create_index('idx_data_files_status', 'data_files', ['status'])
    op.create_index('idx_data_files_upload_time', 'data_files', ['upload_time'])
    
    # Create temporal_alignments table
    op.create_table('temporal_alignments',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('data_file_id', sa.Integer(), nullable=False),
        sa.Column('reference_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sequence_order', sa.Integer(), nullable=False),
        sa.Column('alignment_confidence', sa.Float(), default=1.0),
        sa.Column('column_mapping', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('schema_compatibility', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('created_time', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['data_file_id'], ['data_files.id'], ),
    )
    
    # Create indexes for temporal_alignments
    op.create_index('idx_temporal_alignments_file_id', 'temporal_alignments', ['data_file_id'])
    op.create_index('idx_temporal_alignments_date', 'temporal_alignments', ['reference_date'])
    
    # Create analysis_sessions table
    op.create_table('analysis_sessions',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('session_uuid', sa.String(length=36), nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_activity', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('context_files', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('conversation_history', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('session_metadata', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('total_queries', sa.Integer(), default=0),
        sa.Column('successful_queries', sa.Integer(), default=0),
    )
    
    # Create indexes for analysis_sessions
    op.create_index('idx_analysis_sessions_uuid', 'analysis_sessions', ['session_uuid'])
    op.create_index('idx_analysis_sessions_activity', 'analysis_sessions', ['last_activity'])
    
    # Create chat_messages table
    op.create_table('chat_messages',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('message_type', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('graph_execution_id', sa.String(length=100)),
        sa.Column('graph_state', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('node_outputs', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('analysis_intent', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('related_files', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('analysis_results', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('processing_time', sa.Float()),
        sa.ForeignKeyConstraint(['session_id'], ['analysis_sessions.id'], ),
    )
    
    # Create indexes for chat_messages
    op.create_index('idx_chat_messages_session', 'chat_messages', ['session_id'])
    op.create_index('idx_chat_messages_timestamp', 'chat_messages', ['timestamp'])
    op.create_index('idx_chat_messages_graph_execution', 'chat_messages', ['graph_execution_id'])
    
    # Create analysis_results table
    op.create_table('analysis_results',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('result_uuid', sa.String(length=36), nullable=False, unique=True),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('data_file_id', sa.Integer()),
        sa.Column('analysis_type', sa.Enum(name='analysistype'), nullable=False),
        sa.Column('operation_name', sa.String(length=100), nullable=False),
        sa.Column('parameters', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('involved_files', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('time_range', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('result_data', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('derived_insights', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('recommended_actions', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('confidence_score', sa.Float(), default=0.0),
        sa.Column('generated_code', sa.Text()),
        sa.Column('execution_log', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('computation_time', sa.Float(), nullable=False),
        sa.Column('cache_key', sa.String(length=200)),
        sa.Column('is_cached', sa.Boolean(), default=False),
        sa.ForeignKeyConstraint(['session_id'], ['analysis_sessions.id'], ),
        sa.ForeignKeyConstraint(['data_file_id'], ['data_files.id'], ),
    )
    
    # Create indexes for analysis_results
    op.create_index('idx_analysis_results_session', 'analysis_results', ['session_id'])
    op.create_index('idx_analysis_results_type', 'analysis_results', ['analysis_type'])
    op.create_index('idx_analysis_results_cache', 'analysis_results', ['cache_key'])
    op.create_index('idx_analysis_results_created', 'analysis_results', ['created_at'])
    
    # Create background_tasks table
    op.create_table('background_tasks',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('task_uuid', sa.String(length=36), nullable=False, unique=True),
        sa.Column('session_id', sa.Integer()),
        sa.Column('task_type', sa.String(length=50), nullable=False),
        sa.Column('task_name', sa.String(length=100), nullable=False),
        sa.Column('parameters', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('status', sa.Enum(name='taskstatus'), default='pending'),
        sa.Column('progress', sa.Float(), default=0.0),
        sa.Column('status_message', sa.String(length=500)),
        sa.Column('result_data', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('error_message', sa.Text()),
        sa.Column('logs', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('celery_task_id', sa.String(length=100), unique=True),
        sa.ForeignKeyConstraint(['session_id'], ['analysis_sessions.id'], ),
    )
    
    # Create indexes for background_tasks
    op.create_index('idx_background_tasks_status', 'background_tasks', ['status'])
    op.create_index('idx_background_tasks_type', 'background_tasks', ['task_type'])
    op.create_index('idx_background_tasks_celery', 'background_tasks', ['celery_task_id'])
    
    # Create graph_executions table
    op.create_table('graph_executions',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('execution_uuid', sa.String(length=36), nullable=False, unique=True),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('graph_name', sa.String(length=100), nullable=False),
        sa.Column('input_query', sa.Text(), nullable=False),
        sa.Column('current_node', sa.String(length=50)),
        sa.Column('completed_nodes', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('node_outputs', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('graph_state', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('final_result', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('execution_path', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('status', sa.Enum(name='taskstatus'), default='pending'),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('total_time', sa.Float()),
        sa.Column('error_node', sa.String(length=50)),
        sa.Column('error_message', sa.Text()),
        sa.Column('retry_count', sa.Integer(), default=0),
        sa.ForeignKeyConstraint(['session_id'], ['analysis_sessions.id'], ),
    )
    
    # Create indexes for graph_executions
    op.create_index('idx_graph_executions_session', 'graph_executions', ['session_id'])
    op.create_index('idx_graph_executions_status', 'graph_executions', ['status'])
    op.create_index('idx_graph_executions_started', 'graph_executions', ['started_at'])
    
    # Create query_cache table
    op.create_table('query_cache',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('cache_key', sa.String(length=200), nullable=False, unique=True),
        sa.Column('query_hash', sa.String(length=64), nullable=False),
        sa.Column('parameters_hash', sa.String(length=64), nullable=False),
        sa.Column('result_data', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('result_metadata', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_accessed', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('access_count', sa.Integer(), default=0),
        sa.Column('ttl', sa.Integer(), default=3600),
    )
    
    # Create indexes for query_cache
    op.create_index('idx_query_cache_key', 'query_cache', ['cache_key'])
    op.create_index('idx_query_cache_hash', 'query_cache', ['query_hash'])
    op.create_index('idx_query_cache_accessed', 'query_cache', ['last_accessed'])
    
    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('labels', postgresql.JSON(astext_type=sa.Text())),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    # Create indexes for system_metrics
    op.create_index('idx_system_metrics_name', 'system_metrics', ['metric_name'])
    op.create_index('idx_system_metrics_timestamp', 'system_metrics', ['timestamp'])


def downgrade():
    """Drop all V2 tables"""
    op.drop_table('system_metrics')
    op.drop_table('query_cache')
    op.drop_table('graph_executions')
    op.drop_table('background_tasks')
    op.drop_table('analysis_results')
    op.drop_table('chat_messages')
    op.drop_table('analysis_sessions')
    op.drop_table('temporal_alignments')
    op.drop_table('data_files')
    
    # Drop ENUM types
    op.execute("DROP TYPE taskstatus")
    op.execute("DROP TYPE analysistype")
    op.execute("DROP TYPE filestatus")
    op.execute("DROP TYPE timegranularity")