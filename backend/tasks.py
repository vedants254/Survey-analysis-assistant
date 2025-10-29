import asyncio
import logging
from celery_app import celery_app
from typing import Dict, Any

from langgraph_workflow import analysis_workflow, AnalysisState, register_progress_callback, unregister_progress_callback
from app.db import get_db, AsyncSession
from models.v2_models import AnalysisResult, ChatMessage, AnalysisType

logger = logging.getLogger(__name__)

progress_store = {}

async def update_task_progress(task_id: str, progress_data: Dict[str, Any]):
    """Mock function to simulate sending progress updates."""
    logger.info(f"[TASK_ID: {task_id}] Progress: {progress_data.get('progress_percentage')}% - {progress_data.get('status_message')}")
    progress_store[task_id] = progress_data

@celery_app.task(bind=True, time_limit=600, soft_time_limit=580)
def execute_workflow_task(self, task_id: str, analysis_request_data: Dict[str, Any], request_id: str = None):
    """Celery task to execute the clean LangGraph workflow."""
    from celery_app import request_id_var
    if request_id:
        request_id_var.set(request_id)
    logger.info(f"[CELERY_TASK: {task_id}] Starting workflow execution.")

    from routers.v2_analytics import MultiTemporalAnalysisRequest
    analysis_request = MultiTemporalAnalysisRequest(**analysis_request_data)
    session_id = analysis_request.session_id

    loop = asyncio.get_event_loop()

    async def run_workflow():
        register_progress_callback(task_id, lambda progress: update_task_progress(task_id, progress))

        initial_state = {
            "query": analysis_request.query,
            "files": [{"file_id": fid} for fid in analysis_request.file_ids],
            "session_id": session_id,
            "execution_id": task_id,
            "current_node": "initializing",
            "completed_nodes": [],
            "node_outputs": {},
            "errors": [],
            "parsed_files": [],
            "schema_aligned": False,
            "common_columns": [],
            "operation_type": "unknown",
            "analysis_plan": {},
            "target_metrics": [],
            "time_dimension": "month",
            "aligned_data": {},
            "generated_code": "",
            "validated_code": "",
            "execution_results": {},
            "trends": [],
            "patterns": [],
            "anomalies": [],
            "final_result": {},
            "insights": [],
            "recommended_actions": [],
            "confidence_score": 0.0
        }

        workflow_config = {"configurable": {"thread_id": task_id}}
        final_state = await analysis_workflow.ainvoke(initial_state, config=workflow_config)

        final_result = final_state.get("final_result", {})

        if not final_result:
            raise ValueError("Workflow completed but did not produce a final result.")

        db_session_gen = get_db()
        db: AsyncSession = await anext(db_session_gen)
        try:
            new_analysis = AnalysisResult(
                session_id=session_id,
                analysis_type=AnalysisType.TEMPORAL_COMPARISON,
                operation_name=final_result.get("operation_type", "comprehensive_analysis"),
                parameters={"query": analysis_request.query, "file_ids": analysis_request.file_ids},
                involved_files=analysis_request.file_ids,
                result_data=final_result,
                computation_time=final_result.get("processing_time", 0.0)
            )
            db.add(new_analysis)

            assistant_message = ChatMessage(
                session_id=session_id,
                message_type="assistant",
                content=f"**Comprehensive Analysis Completed**\n\n{final_result.get('executive_summary', 'Analysis completed successfully.')}",
                analysis_results={
                    "analysis_type": "comprehensive",
                    "task_id": task_id
                }
            )
            db.add(assistant_message)
            await db.commit()
            logger.info(f"[CELERY_TASK: {task_id}] Successfully saved analysis results to database.")
        finally:
            await db.close()

        unregister_progress_callback(task_id)
        return final_result

    try:
        result = loop.run_until_complete(run_workflow())
        logger.info(f"[CELERY_TASK: {task_id}] Workflow finished successfully.")
        return result
    except Exception as e:
        logger.error(f"[CELERY_TASK: {task_id}] Workflow execution failed: {e}", exc_info=True)
        raise
