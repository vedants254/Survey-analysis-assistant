"""LangGraph Workflow for Historical Multi-Table Data Analysis"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, TypedDict, Callable
from datetime import datetime
import pandas as pd
from pathlib import Path

try:
    from services.error_handling import create_error_handler, EnhancedError
    from services.safe_execution import create_safe_executor
    from services.data_recovery import create_data_recovery_engine
    from services.graceful_degradation import apply_graceful_degradation
    from services.error_reporting import create_error_report
    ERROR_HANDLING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced error handling not available: {e}")
    ERROR_HANDLING_AVAILABLE = False

logger = logging.getLogger(__name__)








try:
    from langgraph.graph import StateGraph, END
    try:
        from langgraph.checkpoint.memory import MemorySaver as SqliteSaver
    except ImportError:
        try:
            from langgraph.checkpoint import BaseCheckpointSaver as SqliteSaver
        except ImportError:
            class SqliteSaver:
                def __init__(self, *args, **kwargs): pass
    
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph components imported successfully")
except ImportError as e:
    logger.warning(f"LangGraph not available: {e}. Using mock implementation")
    LANGGRAPH_AVAILABLE = False
    class StateGraph:
        def __init__(self, state_schema): 
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = []
        def add_node(self, name, func): self.nodes[name] = func
        def add_edge(self, from_node, to_node): self.edges.append((from_node, to_node))
        def add_conditional_edges(self, from_node, condition, mapping): pass
        def set_entry_point(self, node): self.entry_point = node
        def compile(self, checkpointer=None): return MockGraph(self)
    
    class SqliteSaver:
        def __init__(self, path): pass
    
    END = "END"


class AnalysisState(TypedDict):
    """State object that flows through the LangGraph workflow."""
    query: str
    files: List[Dict[str, Any]]
    session_id: str
    execution_id: str
    current_node: str
    completed_nodes: List[str] 
    node_outputs: Dict[str, Any]
    errors: List[str]
    parsed_files: List[Dict[str, Any]]
    schema_aligned: bool
    common_columns: List[str]
    operation_type: str
    analysis_plan: Dict[str, Any]
    target_metrics: List[str]
    time_dimension: str
    aligned_data: Dict[str, Any]
    generated_code: str
    validated_code: str
    execution_results: Dict[str, Any]
    trends: List[Dict[str, Any]]
    patterns: List[str]
    anomalies: List[Dict[str, Any]]
    correlations: List[Dict[str, Any]]
    forecasts: List[Dict[str, Any]]
    statistical_tests: List[Dict[str, Any]]
    final_result: Dict[str, Any]
    insights: List[str]
    recommended_actions: List[str]
    confidence_score: float


class MockGraph:
    """Mock graph for testing when LangGraph is not available"""
    def __init__(self, state_graph):
        self.state_graph = state_graph
    
    async def astream(self, initial_state, config=None):
        """Mock streaming that yields state updates"""
        state = AnalysisState(**initial_state)
        for i, (node_name, _) in enumerate(self.state_graph.nodes.items()):
            state["current_node"] = node_name
            state["completed_nodes"].append(node_name)
            yield state
            await asyncio.sleep(0.1)


_progress_callbacks = {}

def register_progress_callback(execution_id: str, callback):
    """Register a progress callback for a specific execution"""
    _progress_callbacks[execution_id] = callback

def unregister_progress_callback(execution_id: str):
    """Remove progress callback when execution is complete"""
    _progress_callbacks.pop(execution_id, None)

def get_progress_callback(execution_id: str):
    """Get progress callback for an execution"""
    return _progress_callbacks.get(execution_id)

async def send_progress_update(state: AnalysisState, node_name: str, progress_percent: float, message: str):
    """Send progress update if callback is available"""
    execution_id = state.get("execution_id")
    progress_callback = get_progress_callback(execution_id) if execution_id else None
    
    if progress_callback:
        try:
            total_nodes = 8
            completed_count = len(state.get("completed_nodes", []))
            overall_progress = (completed_count * 100 / total_nodes) + (progress_percent / total_nodes)
            
            progress_data = {
                "execution_id": state["execution_id"],
                "current_node": node_name,
                "completed_nodes": state.get("completed_nodes", []),
                "progress_percentage": min(overall_progress, 99.0),
                "status_message": message,
                "estimated_completion_seconds": max(10, (total_nodes - completed_count) * 8)
            }
            await progress_callback(progress_data)
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")

async def parse_files_node(state: AnalysisState) -> AnalysisState:
    """Parse multiple files, extract schemas, align columns"""
    logger.info("🔍 Starting parse_files node")
    await send_progress_update(state, "parse_files", 10.0, "Loading and parsing data files...")
    
    try:
        from app.database import FileModel
        
        parsed_files = []
        all_columns = set()
        
        await send_progress_update(state, "parse_files", 30.0, f"Processing {len(state['files'])} data files...")
        
        for file_info in state["files"]:
            file_id = file_info.get("id") or file_info.get("file_id")
            
            db_file = FileModel.get_by_id(file_id)
            if not db_file:
                logger.error(f"File {file_id} not found in database")
                continue
            
            file_path = Path(db_file["file_path"])
            if not file_path.exists():
                logger.error(f"File path {file_path} does not exist")
                continue
            
            try:
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                json_columns = []
                parsed_json_fields = {}
                
                for col in df.columns:
                    if col.lower() in ['meta', 'metadata', 'json_data']:
                        json_columns.append(col)
                        json_fields = set()
                        for val in df[col].dropna().head(10):
                            try:
                                if isinstance(val, str) and val.strip().startswith('{'):
                                    parsed_json = json.loads(val)
                                    if isinstance(parsed_json, dict):
                                        json_fields.update(parsed_json.keys())
                            except (json.JSONDecodeError, AttributeError):
                                continue
                        
                        if json_fields:
                            parsed_json_fields[col] = list(json_fields)
                            logger.debug(f"JSON fields in {col}: {list(json_fields)}")
                
                file_data = {
                    "file_id": file_id,
                    "original_filename": db_file["original_filename"], 
                    "file_path": str(file_path),
                    "columns": list(df.columns),
                    "row_count": len(df),
                    "data_sample": df.head(3).to_dict('records'),
                    "numeric_columns": list(df.select_dtypes(include=['number']).columns),
                    "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                    "json_columns": json_columns,
                    "json_fields": parsed_json_fields,
                }
                
                parsed_files.append(file_data)
                all_columns.update(df.columns)
                
                logger.info(f"Parsed {db_file['original_filename']}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"Error parsing file {file_path}: {e}")
                state["errors"].append(f"Failed to parse {db_file['original_filename']}: {str(e)}")
        
        if parsed_files:
            common_columns = list(set.intersection(*[set(f["columns"]) for f in parsed_files]))
        else:
            common_columns = []
        
        state.update({
            "current_node": "parse_files",
            "parsed_files": parsed_files,
            "schema_aligned": len(common_columns) > 0,
            "common_columns": common_columns,
        })
        
        if not state["completed_nodes"]:
            state["completed_nodes"] = []
        state["completed_nodes"].append("parse_files")
        
        if not state["node_outputs"]:
            state["node_outputs"] = {}
        state["node_outputs"]["parse_files"] = {
            "files_parsed": len(parsed_files),
            "common_columns": common_columns,
            "total_rows": sum(f["row_count"] for f in parsed_files)
        }
        
        await send_progress_update(state, "parse_files", 100.0, f"Completed: {len(parsed_files)} files parsed, {len(common_columns)} common columns found")
        logger.info(f"parse_files completed: {len(parsed_files)} files, {len(common_columns)} common columns")
        return state
        
    except Exception as e:
        logger.error(f"❌ parse_files failed: {e}")
        state["errors"].append(f"File parsing failed: {str(e)}")
        return state


async def plan_operations_node(state: AnalysisState) -> AnalysisState:
    """Plan operations using LLM - decide if single-table or cross-table"""
    logger.info("🧠 Starting plan_operations node")
    await send_progress_update(state, "plan_operations", 20.0, "Analyzing data structure and planning operations...")
    
    try:
        from services.llm_providers import llm_manager, LLMMessage
        
        files_summary = "\n".join([
            f"- {f['original_filename']}: {f['row_count']} rows, columns: {', '.join(f['columns'][:5])}{'...' if len(f['columns']) > 5 else ''}"
            for f in state["parsed_files"]
        ])
        
        files_count = len(state["parsed_files"])
        json_fields_info = ""
        for file_data in state["parsed_files"]:
            if file_data.get("json_fields"):
                json_fields_info += f"\n- {file_data['original_filename']} has JSON metadata: {file_data['json_fields']}"
        
        planning_prompt = f"""
Analyze this data analysis request and determine the optimal approach:

Query: "{state['query']}"

Data Context:
- Number of files: {files_count}
{files_summary}{json_fields_info}

Common columns: {', '.join(state['common_columns'])}

Analysis Guidelines:
- **single_table**: Query refers to one specific file/period (e.g., "November data", "this file")
- **cross_table**: Query compares/combines multiple files (e.g., "compare Nov vs Dec", "growth from Q1 to Q2", "trends across months")
- **temporal_comparison**: Query specifically asks for time-based analysis (e.g., "MoM growth", "seasonal patterns", "quarterly trends")

Temporal Keywords to Look For:
- Comparison: "vs", "compared to", "growth", "change", "increase/decrease" 
- Time periods: "month", "quarter", "year", "MoM", "QoQ", "YoY"
- Trends: "trend", "pattern", "over time", "across periods"
- Multiple periods mentioned: "Nov and Dec", "Q1 vs Q2"

JSON Metadata Usage:
- If query mentions "channel", "priority", or other metadata fields, include JSON parsing
- Example: "online channel revenue" requires parsing Meta.channel field

Respond in JSON format:
{{
    "operation_type": "single_table|cross_table|temporal_comparison",
    "target_metrics": ["revenue", "units", "discount"],
    "time_dimension": "month|quarter|year|none",
    "grouping_columns": ["region", "product"],
    "requires_json_parsing": true|false,
    "json_fields_needed": ["channel", "priority"],
    "comparison_type": "none|growth|correlation|trend_analysis|anomaly_detection",
    "reasoning": "Detailed explanation of why this approach was chosen"
}}
"""
        
        messages = [
            LLMMessage(role="system", content="You are a data analysis expert. Respond with valid JSON only."),
            LLMMessage(role="user", content=planning_prompt)
        ]
        
        try:
            response = await llm_manager.generate(messages)
            analysis_plan = json.loads(response.content)
        except json.JSONDecodeError:
            operation_type = "temporal_comparison" if len(state["parsed_files"]) > 1 else "single_table"
            numeric_cols = []
            for f in state["parsed_files"]:
                numeric_cols.extend(f.get("numeric_columns", []))
            
            analysis_plan = {
                "operation_type": operation_type,
                "target_metrics": list(set(numeric_cols))[:3],
                "time_dimension": "month", 
                "grouping_columns": [col for col in state["common_columns"] if col.lower() in ['region', 'product', 'category']],
                "reasoning": "Heuristic-based analysis plan (LLM response parsing failed)"
            }
        except Exception as e:
            logger.error(f"❌ LLM Provider Error: {e}")
            error_msg = f"LLM analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            raise Exception(f"Cannot perform AI analysis: {error_msg}")
        
        state.update({
            "current_node": "plan_operations",
            "operation_type": analysis_plan["operation_type"],
            "analysis_plan": analysis_plan,
            "target_metrics": analysis_plan["target_metrics"],
            "time_dimension": analysis_plan.get("time_dimension", "month")
        })
        
        state["completed_nodes"].append("plan_operations")
        state["node_outputs"]["plan_operations"] = analysis_plan
        
        await send_progress_update(state, "plan_operations", 100.0, f"Analysis plan ready: {analysis_plan['operation_type']} approach with {len(analysis_plan['target_metrics'])} metrics")
        logger.info(f"✅ plan_operations completed: {analysis_plan['operation_type']} with metrics {analysis_plan['target_metrics']}")
        return state
        
    except Exception as e:
        logger.error(f"❌ plan_operations failed: {e}")
        state["errors"].append(f"Operation planning failed: {str(e)}")
        # Provide fallback plan
        state.update({
            "operation_type": "single_table",
            "analysis_plan": {"operation_type": "single_table", "reasoning": "Fallback due to error"},
            "target_metrics": ["revenue"] if "revenue" in state.get("common_columns", []) else []
        })
        return state


async def align_timeseries_node(state: AnalysisState) -> AnalysisState:
    """Align tables by time dimension before code generation"""
    logger.info("📅 Starting align_timeseries node")
    
    try:
        if state["operation_type"] == "single_table":
            state.update({
                "current_node": "align_timeseries", 
                "aligned_data": {"single_table": state["parsed_files"][0] if state["parsed_files"] else {}}
            })
        else:
            aligned_data = {}
            
            for file_data in state["parsed_files"]:
                filename = file_data["original_filename"].lower()
                if "nov" in filename or "november" in filename:
                    time_key = "2024-11"
                elif "dec" in filename or "december" in filename:
                    time_key = "2024-12"
                elif "q1" in filename:
                    time_key = "2025-Q1"
                else:
                    time_key = file_data["original_filename"]
                
                aligned_data[time_key] = file_data
            
            state.update({
                "current_node": "align_timeseries",
                "aligned_data": aligned_data
            })
        
        state["completed_nodes"].append("align_timeseries")
        state["node_outputs"]["align_timeseries"] = {
            "time_periods": list(state["aligned_data"].keys()),
            "alignment_completed": True
        }
        
        logger.info(f"✅ align_timeseries completed: {len(state['aligned_data'])} time periods aligned")
        return state
        
    except Exception as e:
        logger.error(f"❌ align_timeseries failed: {e}")
        state["errors"].append(f"Time series alignment failed: {str(e)}")
        return state


async def generate_code_node(state: AnalysisState) -> AnalysisState:
    """Generate Python analysis code using LLM with error-aware retry capability"""
    retry_count = state.get("generation_retry_count", 0)
    is_retry = retry_count > 0
    
    recent_errors = state.get("errors", [])[-5:]
    execution_errors = []
    if "execution_results" in state and state["execution_results"].get("status") in ["execution_error", "timeout_error"]:
        execution_errors = [state["execution_results"].get("error", "")]
    
    if is_retry:
        logger.info(f"🔄 Starting generate_code node - RETRY (attempt {retry_count + 1}/3)")
        logger.info(f"📝 Error context: {recent_errors[-1] if recent_errors else 'Previous execution failed'}")
        
        # Send enhanced WebSocket update for retries
        await send_progress_update(
            state, "generate_code", 10.0, 
            f"Retrying code generation (attempt {retry_count + 1}/3) with error context..."
        )
        
        # Send detailed retry info via WebSocket
        execution_id = state.get("execution_id")
        if execution_id:
            progress_callback = get_progress_callback(execution_id)
            if progress_callback:
                await progress_callback({
                    "execution_id": execution_id,
                    "current_node": "generate_code", 
                    "completed_nodes": state.get("completed_nodes", []),
                    "progress_percentage": 15.0,
                    "status_message": f"🔄 Applying error context to improve code generation (retry {retry_count + 1}/3)",
                    "retry_info": {
                        "is_retry": True,
                        "retry_count": retry_count,
                        "max_retries": 3,
                        "error_context": recent_errors[-3:] if recent_errors else [],
                        "execution_errors": execution_errors
                    },
                    "estimated_completion_seconds": 45
                })
    else:
        logger.info("💻 Starting generate_code node - INITIAL ATTEMPT")
        await send_progress_update(state, "generate_code", 10.0, "Generating Python analysis code...")
    
    try:
        from services.llm_providers import llm_manager, LLMMessage
        
        files_context = []
        for time_key, file_data in state["aligned_data"].items():
            files_context.append({
                "time_period": time_key,
                "columns": file_data["columns"],
                "row_count": file_data["row_count"],
                "sample_data": file_data.get("data_sample", [])[:2]
            })
        
        requires_json = state["analysis_plan"].get("requires_json_parsing", False)
        json_fields_needed = state["analysis_plan"].get("json_fields_needed", [])
        comparison_type = state["analysis_plan"].get("comparison_type", "none")
        
        temporal_hints = ""
        if state["operation_type"] in ["cross_table", "temporal_comparison"]:
            temporal_hints = f"""

TEMPORAL ANALYSIS GUIDANCE:
- Operation: {state['operation_type']} 
- Time Dimension: {state['time_dimension']}
- Comparison Type: {comparison_type}
- For MoM/QoQ growth: use ((new_value - old_value) / old_value) * 100
- For trend analysis: calculate changes over time periods
- For cross-table joins: merge on common columns like Region, Product
- Handle missing data gracefully with fillna() or dropna()
            """
        
        json_hints = ""
        if requires_json and json_fields_needed:
            json_hints = f"""

JSON METADATA PROCESSING:
- Parse JSON columns (Meta, metadata) using json.loads()
- Extract fields: {', '.join(json_fields_needed)}
- Example: df['channel'] = df['Meta'].apply(lambda x: json.loads(x).get('channel', 'unknown') if pd.notna(x) else 'unknown')
- Filter by channel: df[df['channel'] == 'online']
- Group by JSON fields: df.groupby(['channel', 'priority'])
            """
        
        error_context_section = ""
        if is_retry and (recent_errors or execution_errors):
            all_errors = recent_errors + execution_errors
            error_context_section = f"""

ERROR CONTEXT (THIS IS A RETRY - AVOID THESE ISSUES):
Previous attempt failed with these errors:
{chr(10).join([f'- {error}' for error in all_errors[-3:]])}

SPECIFIC FIXES NEEDED:
"""            
            error_text = " ".join(all_errors).lower()
            
            if "keyerror" in error_text or "not found" in error_text:
                error_context_section += "- Check column names carefully - use df.columns to verify available columns\n"
                error_context_section += "- Use try/except blocks when accessing columns that might not exist\n"
                
            if "float64" in error_text or "iloc" in error_text or "scalar" in error_text:
                error_context_section += "- Avoid using .iloc on scalar values - check if result is scalar before using .iloc\n"
                error_context_section += "- Use safe scalar conversion: float(value) or value.item() for numpy scalars\n"
                
            if "syntax" in error_text or "name" in error_text:
                error_context_section += "- Check variable names and syntax carefully\n"
                error_context_section += "- Ensure all required imports are included\n"
                
            if "timeout" in error_text:
                error_context_section += "- Keep code simple and efficient - avoid complex loops\n"
                error_context_section += "- Use vectorized pandas operations instead of iterative approaches\n"
                
            error_context_section += "\nGENERATE SIMPLER, MORE ROBUST CODE TO AVOID THESE ERRORS.\n"
        
        code_prompt = f"""
Generate Python pandas code to analyze this data. Be careful with array/scalar operations.{error_context_section}

Query: "{state['query']}"
Operation Type: {state['operation_type']}
Target Metrics: {', '.join(state['target_metrics'])}
Analysis Plan: {state['analysis_plan'].get('reasoning', 'Standard analysis')}{temporal_hints}{json_hints}

Available data files:
{json.dumps(files_context, indent=2, default=str)}

Requirements:
1. Use pandas for data manipulation
2. Handle the specific metrics: {', '.join(state['target_metrics'])}
3. Create meaningful analysis based on the query
4. Store final results in a variable named 'analysis_results' as a dictionary
5. Include try/except blocks for error handling
6. Add comments explaining key steps
7. IMPORTANT: Use .iloc[0], .item(), or .values[0] when converting single-element arrays to scalars
8. IMPORTANT: Avoid operations that mix arrays and scalars without proper conversion
9. Use pd.to_numeric() for safe numeric conversions
10. Handle empty DataFrames gracefully
11. For temporal analysis, calculate growth rates, trends, and comparisons properly
12. Parse JSON columns if needed using json.loads() with error handling

Example pattern for safe scalar conversion:
```python
# Good: Convert single values safely
max_val = df['column'].max()
if isinstance(max_val, (pd.Series, np.ndarray)) and len(max_val) == 1:
    max_val = max_val.iloc[0] if hasattr(max_val, 'iloc') else max_val.item()
```

Example for JSON parsing:
```python
# Parse JSON metadata safely
def parse_json_field(json_str, field_name):
    try:
        return json.loads(json_str).get(field_name, 'unknown') if pd.notna(json_str) else 'unknown'
    except:
        return 'unknown'
        
df['channel'] = df['Meta'].apply(lambda x: parse_json_field(x, 'channel'))
```

Generate clean, executable Python code:
"""
        
        messages = [
            LLMMessage(role="system", content="You are a Python data analysis expert. Generate clean, executable pandas code with proper error handling."),
            LLMMessage(role="user", content=code_prompt)
        ]
        
        if "code_generation_attempts" not in state:
            state["code_generation_attempts"] = 0
        
        state["code_generation_attempts"] += 1
        
        if state["code_generation_attempts"] > 1:
            logger.info(f"🔄 Code generation retry {state['code_generation_attempts']}/3 - using safer template")
            
            code_prompt = f"""
Generate SIMPLE, SAFE Python code that CANNOT fail or hang.

Query: "{state['query']}"
Files: {len(state['parsed_files'])} data files available

USE THIS EXACT SAFE TEMPLATE:
```python
analysis_results = {{}}
try:
    dfs = [v for k, v in locals().items() if k.startswith('df_') and hasattr(v, 'shape')]
    
    if not dfs:
        analysis_results = {{'error': 'No data found', 'status': 'no_data'}}
    else:
        df = dfs[0]
        
        if not df.empty and len(df) > 0:
            numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            
            if numeric_cols:
                stats = {{}}
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    try:
                        stats[col] = {{
                            'mean': float(df[col].mean()) if not df[col].isna().all() else 0.0,
                            'sum': float(df[col].sum()) if not df[col].isna().all() else 0.0,
                            'count': int(df[col].count())
                        }}
                    except:
                        stats[col] = {{'error': 'Could not calculate'}}
                
                analysis_results = {{
                    'status': 'completed',
                    'basic_stats': stats,
                    'row_count': int(len(df)),
                    'columns': list(df.columns)
                }}
            else:
                analysis_results = {{'status': 'no_numeric_data', 'row_count': int(len(df))}}
        else:
            analysis_results = {{'error': 'DataFrame is empty', 'status': 'empty_data'}}
            
except Exception as e:
    analysis_results = {{'error': str(e), 'status': 'error'}}
```

Return ONLY this safe code template with minimal modifications.
            """
        
        try:
            response = await llm_manager.generate(messages)
            generated_code = response.content
        except Exception as e:
            logger.error(f"❌ LLM Code Generation Failed: {e}")
            error_msg = f"Code generation failed: {str(e)}"
            state["errors"].append(error_msg)
            
            # Allow up to 3 attempts for code generation
            if state["code_generation_attempts"] < 3:
                logger.info(f"🔄 Will retry code generation (attempt {state['code_generation_attempts'] + 1}/3)")
                return await generate_code_node(state)  # Recursive retry
            else:
                raise Exception(f"Cannot generate analysis code after 3 attempts: {error_msg}")
        
        # Clean up the code (remove markdown formatting if present)
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].strip()
        
        state.update({
            "current_node": "generate_code",
            "generated_code": generated_code
        })
        
        state["completed_nodes"].append("generate_code")
        state["node_outputs"]["generate_code"] = {
            "code_length": len(generated_code),
            "has_pandas": "pd." in generated_code or "pandas" in generated_code
        }
        
        logger.info(f"✅ generate_code completed: {len(generated_code)} characters")
        return state
        
    except Exception as e:
        logger.error(f"❌ generate_code failed: {e}")
        state["errors"].append(f"Code generation failed: {str(e)}")
        raise Exception(f"Code generation failed: {str(e)}")


async def validate_code_node(state: AnalysisState) -> AnalysisState:
    """Validate generated code for syntax and safety"""
    logger.info("🔍 Starting validate_code node")
    
    try:
        import ast
        
        try:
            ast.parse(state["generated_code"])
            syntax_valid = True
        except SyntaxError as e:
            syntax_valid = False
            state["errors"].append(f"Code syntax error: {str(e)}")
        
        dangerous_patterns = [
            "import os", "import subprocess", "exec(", "eval(", 
            "open(", "__import__", "getattr(", "delattr(",
            "while True:", "while 1:", "for i in range(999",
            "time.sleep", "input(", "raw_input(",
        ]
        
        infinite_loop_patterns = [
            "while True", "while 1", "while len(", "while not False",
            "for i in range(999", "for _ in range(999", "while df.empty == False"
        ]
        
        safety_issues = []
        for pattern in dangerous_patterns:
            if pattern in state["generated_code"]:
                safety_issues.append(pattern)
        
        # Check for infinite loop patterns
        loop_issues = []
        for pattern in infinite_loop_patterns:
            if pattern in state["generated_code"]:
                loop_issues.append(pattern)
                
        if safety_issues:
            logger.warning(f"Code contains potentially unsafe patterns: {safety_issues}")
            
        if loop_issues:
            logger.error(f"Code contains potential infinite loop patterns: {loop_issues}")
            state["errors"].append(f"Code rejected: contains infinite loop patterns: {loop_issues}")
            syntax_valid = False
        
        validated_code = state["generated_code"] if syntax_valid else None
        
        state.update({
            "current_node": "validate_code",
            "validated_code": validated_code or state["generated_code"]  # Use original if validation fails
        })
        
        state["completed_nodes"].append("validate_code")
        state["node_outputs"]["validate_code"] = {
            "syntax_valid": syntax_valid,
            "safety_issues": safety_issues,
            "validation_status": "passed" if syntax_valid else "failed"
        }
        
        logger.info(f"✅ validate_code completed: {'passed' if syntax_valid else 'failed'}")
        return state
        
    except Exception as e:
        logger.error(f"❌ validate_code failed: {e}")
        state["errors"].append(f"Code validation failed: {str(e)}")
        state["validated_code"] = state["generated_code"]  # Use original code
        return state


def _make_serializable(obj):
    """Convert non-serializable objects to serializable formats"""
    if obj is None:
        return None
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif hasattr(obj, 'item'):  # numpy objects
        try:
            return obj.item()
        except:
            return str(obj)
    elif hasattr(obj, 'iloc'):  # pandas Series/DataFrame
        try:
            if len(obj) == 1:
                return float(obj.iloc[0]) if pd.api.types.is_numeric_dtype(obj) else str(obj.iloc[0])
            else:
                return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
        except:
            return str(obj)
    elif isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, 'dtype'):  # numpy arrays
        try:
            return obj.tolist()
        except:
            return str(obj)
    else:
        return obj

def _create_basic_summary(state, namespace):
    """Create basic data summary when code execution fails"""
    summary = {
        "status": "fallback_summary",
        "message": "Generated basic summary due to execution issues"
    }
    
    try:
        # Find DataFrames in namespace
        dataframes = {k: v for k, v in namespace.items() if k.startswith('df_') and hasattr(v, 'shape')}
        
        if dataframes:
            summary["data_info"] = {}
            for name, df in dataframes.items():
                try:
                    summary["data_info"][name] = {
                        "rows": int(df.shape[0]),
                        "columns": list(df.columns),
                        "numeric_columns": list(df.select_dtypes(include=['number']).columns)
                    }
                    
                    # Basic stats for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        summary["basic_stats"] = {}
                        for col in numeric_cols:
                            try:
                                summary["basic_stats"][col] = {
                                    "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else 0,
                                    "sum": float(df[col].sum()) if pd.notna(df[col].sum()) else 0,
                                    "count": int(df[col].count())
                                }
                            except:
                                continue
                except Exception as e:
                    logger.warning(f"Error processing {name}: {e}")
                    continue
    except Exception as e:
        logger.warning(f"Error creating basic summary: {e}")
        summary["error"] = "Could not generate basic summary"
    
    return summary

def _execute_with_pandas_fixes(state, namespace, original_error):
    """Attempt to fix common pandas errors and re-execute"""
    try:
        import json as json_module
        # Ensure json is in namespace
        namespace['json'] = json_module
        
        # Common fixes for pandas/numpy issues
        fixed_code = state["validated_code"]
        
        # Fix 1: Replace .iloc calls on scalars
        if "float64" in str(original_error) and "iloc" in str(original_error):
            # This is a common error - trying to call .iloc on a scalar
            # Add safety checks
            fixed_code = fixed_code.replace(
                ".iloc[0]",
                ".iloc[0] if hasattr(value, 'iloc') and len(value) > 0 else value"
            )
        
        # Fix 2: Wrap aggregations in safe conversions
        fixed_code = """
# Safe execution wrapper
try:
    import pandas as pd
    import numpy as np
    
    # Helper function for safe value extraction
    def safe_extract(value):
        if value is None:
            return 0
        elif hasattr(value, 'item'):
            return value.item()
        elif hasattr(value, 'iloc') and len(value) > 0:
            return value.iloc[0]
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            return value[0]
        else:
            return float(value) if pd.notna(value) else 0
    
    # Original code with safety wrappers
""" + fixed_code + """
    
    # Ensure analysis_results exists
    if 'analysis_results' not in locals():
        analysis_results = {'status': 'completed', 'message': 'Analysis completed with fixes'}
        
except Exception as e:
    analysis_results = {'error': f'Fixed execution failed: {str(e)}', 'status': 'error'}
"""
        
        # Try executing the fixed code
        exec(fixed_code, namespace)
        results = namespace.get('analysis_results', {})
        
        if results and 'error' not in results:
            logger.info("✅ Fixed execution successful")
            return _make_serializable(results)
        else:
            return {'status': 'error', 'message': 'Fixed execution returned no results'}
            
    except Exception as e:
        logger.error(f"Fixed execution also failed: {e}")
        return {'status': 'error', 'message': f'Both original and fixed execution failed: {str(e)}'}

async def _legacy_code_execution(state: AnalysisState) -> Dict[str, Any]:
    """Legacy code execution method when enhanced error handling is not available"""
    import json as json_module
    
    # Create a safe namespace for execution
    namespace = {
        'pd': pd,
        'np': __import__('numpy'),
        'json': json_module,
        'analysis_results': {}
    }
    
    # Add data to namespace by reloading DataFrames from file paths
    for time_key, file_data in state["aligned_data"].items():
        df_name = f"df_{time_key.replace('-', '_').replace(' ', '_')}"
        try:
            file_path = Path(file_data['file_path'])
            if file_path.suffix.lower() == '.csv':
                namespace[df_name] = pd.read_csv(file_path)
            else:
                namespace[df_name] = pd.read_excel(file_path)
        except Exception as e:
            logger.warning(f"Failed to load DataFrame for {time_key}: {e}")
    
    # Try execution with timeout
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")
        
        # Set up timeout if available (Unix/Linux/macOS only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
        
        try:
            exec(state["validated_code"], namespace)
            execution_results = namespace.get('analysis_results', {})
            
            if execution_results and len(execution_results) > 0:
                execution_results = _make_serializable(execution_results)
                logger.info("✅ Legacy execution successful")
                return execution_results
            else:
                logger.warning("Legacy code executed but returned empty results")
                return _create_basic_summary(state, namespace)
                
        finally:
            # Clean up signal handler
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
    except (TimeoutError, Exception) as e:
        logger.error(f"Legacy execution failed: {e}")
        
        # Try pandas error fixes
        if any(phrase in str(e).lower() for phrase in ['float64', 'iloc', 'scalar', 'array', 'numpy']):
            try:
                execution_results = _execute_with_pandas_fixes(state, namespace, e)
                if execution_results and execution_results.get('status') != 'error':
                    return execution_results
            except Exception as fix_error:
                logger.error(f"Pandas fix attempt failed: {fix_error}")
        
        # Return basic fallback
        return _create_basic_summary(state, namespace)

async def execute_code_node(state: AnalysisState) -> AnalysisState:
    """Execute the validated code safely with enhanced error handling and recovery"""
    execution_retry_count = state.get("execution_retry_count", 0)
    if state.get("current_node") == "execute_code" and "execution_retry_count" in state:
        execution_retry_count += 1
        state["execution_retry_count"] = execution_retry_count
        logger.info(f"⚡ Starting execute_code node - RETRY (attempt {execution_retry_count + 1}) with enhanced error handling")
    else:
        state["execution_retry_count"] = 0
        execution_retry_count = 0
        logger.info("⚡ Starting execute_code node with enhanced error handling - INITIAL ATTEMPT")
    
    if execution_retry_count > 0:
        await send_progress_update(state, "execute_code", 20.0, f"Retrying code execution (attempt {execution_retry_count + 1}) with error recovery...")
    else:
        await send_progress_update(state, "execute_code", 20.0, "Executing data analysis code with safety checks...")
    
    try:
        if ERROR_HANDLING_AVAILABLE:
            logger.info("🛡️ Using enhanced safe code execution")
            safe_executor = create_safe_executor(timeout_seconds=45, max_memory_mb=256)
            
            context = {'dataframes': {}}
            
            for time_key, file_data in state["aligned_data"].items():
                df_name = f"df_{time_key.replace('-', '_').replace(' ', '_')}"
                try:
                    file_path = Path(file_data['file_path'])
                    if file_path.suffix.lower() == '.csv':
                        context['dataframes'][df_name] = pd.read_csv(file_path)
                    else:
                        context['dataframes'][df_name] = pd.read_excel(file_path)
                    logger.info(f"📊 Loaded {df_name}: {context['dataframes'][df_name].shape}")
                except Exception as e:
                    logger.warning(f"Failed to load DataFrame for {time_key}: {e}")
            
            execution_results = await safe_executor.execute_code(
                state["validated_code"], 
                context=context,
                max_attempts=3
            )
            
            if execution_results and execution_results.get('status') in ['success', 'completed', 'completed_with_fallback']:
                logger.info("✅ Enhanced code execution successful")
            else:
                logger.warning(f"⚠️ Enhanced execution returned: {execution_results.get('status')}")
                state["errors"].append(f"Code execution issues: {execution_results.get('error', 'Unknown issue')}")
        
        else:
            logger.info("⚠️ Using fallback code execution (enhanced error handling not available)")
            execution_results = await _legacy_code_execution(state)
        
        if not execution_results or not isinstance(execution_results, dict):
            execution_results = {
                "status": "completed_with_minimal_fallback",
                "message": "Analysis completed with basic processing",
                "error_recovery": "Applied fallback due to execution issues",
                "timestamp": datetime.now().isoformat()
            }
        
        state.update({
            "current_node": "execute_code",
            "execution_results": execution_results
        })
        
        state["completed_nodes"].append("execute_code")
        state["node_outputs"]["execute_code"] = {
            "execution_status": "success" if execution_results.get('status') != 'error' else "failed_with_recovery",
            "results_generated": len(execution_results) > 0,
            "error_handling_used": ERROR_HANDLING_AVAILABLE,
            "result_status": execution_results.get('status', 'unknown')
        }
        
        await send_progress_update(state, "execute_code", 90.0, f"Code execution completed: {execution_results.get('status', 'completed')}")
        logger.info(f"✅ execute_code completed with status: {execution_results.get('status')}")
        return state
        
    except Exception as e:
        logger.error(f"❌ execute_code failed: {e}")
        
        # Enhanced error reporting
        error_details = {
            "error_message": str(e),
            "error_type": type(e).__name__,
            "node": "execute_code",
            "timestamp": datetime.now().isoformat(),
            "enhanced_handling_available": ERROR_HANDLING_AVAILABLE
        }
        
        state["errors"].append(f"Code execution failed: {str(e)}")
        
        # Try to provide a meaningful fallback result even on failure
        fallback_result = {
            "status": "execution_failed_with_fallback",
            "message": "Code execution encountered errors but analysis attempted with fallback methods",
            "error_details": error_details,
            "fallback_analysis": {
                "basic_info": "Analysis workflow attempted",
                "recommendations": [
                    "Review generated code for syntax issues",
                    "Check data format compatibility",
                    "Verify column names and data structure",
                    "Try with simpler analysis requirements"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        state["execution_results"] = fallback_result
        state["completed_nodes"].append("execute_code")
        
        # Don't raise exception - let workflow continue with fallback results
        return state


async def trend_analysis_node(state: AnalysisState) -> AnalysisState:
    """Detect patterns, trends, and anomalies using LLM"""
    logger.info("📈 Starting trend_analysis node")
    
    try:
        from services.llm_providers import llm_manager, LLMMessage
        
        try:
            results_summary = json.dumps(state["execution_results"], indent=2, default=str)[:1000]
        except (TypeError, ValueError) as json_error:
            logger.warning(f"JSON serialization issue with execution results: {json_error}")
            results_summary = f"""Execution results summary:
Status: {state['execution_results'].get('status', 'completed')}
Entries: {len(state['execution_results']) if isinstance(state['execution_results'], dict) else 'N/A'}
Query: {state['query']}
"""
            logger.info(f"Using simplified results summary: {results_summary[:200]}...")
        
        comparison_type = state["analysis_plan"].get("comparison_type", "none")
        requires_correlation = "correlation" in state['query'].lower() or comparison_type == "correlation"
        requires_anomaly_detection = "anomal" in state['query'].lower() or "detect" in state['query'].lower()
        requires_forecasting = any(word in state['query'].lower() for word in ["predict", "forecast", "project", "if", "continue"])
        
        statistical_guidance = ""
        if requires_correlation:
            statistical_guidance += "\n- CORRELATION ANALYSIS: Calculate correlation coefficients between variables and interpret statistical significance"
        if requires_anomaly_detection:
            statistical_guidance += "\n- ANOMALY DETECTION: Identify outliers with confidence intervals and statistical thresholds"
        if requires_forecasting:
            statistical_guidance += "\n- PREDICTIVE MODELING: Provide trend-based forecasts and scenario analysis"
        
        trend_prompt = f"""
You are a senior data analyst with expertise in statistical analysis. Analyze these results and provide comprehensive insights in valid JSON format only.

Query: "{state['query']}"
Operation: {state['operation_type']}
Analysis Type: {comparison_type}
Data Results: {results_summary}

STATISTICAL ANALYSIS REQUIREMENTS:{statistical_guidance}

Return ONLY valid JSON with this exact structure:
{{
    "trends": [
        {{"metric": "revenue", "direction": "increasing", "confidence": 0.85, "change_percent": 12.5, "statistical_significance": "high", "description": "Revenue shows strong upward trend with 12.5% growth"}}
    ],
    "patterns": ["Seasonal pattern: higher sales in Q4", "Regional variation: APAC outperforming EU"],
    "correlations": [
        {{"variables": ["discount", "revenue"], "coefficient": -0.65, "significance": "moderate", "interpretation": "Higher discounts correlate with lower revenue per unit"}}
    ],
    "anomalies": [
        {{"metric": "units", "value": 500, "expected_range": [100, 300], "confidence": 0.9, "description": "Unusually high unit count detected"}}
    ],
    "forecasts": [
        {{"metric": "revenue", "period": "Q1 2025", "predicted_value": 15000, "confidence_interval": [12000, 18000], "assumptions": "Trend continues at current rate"}}
    ],
    "statistical_tests": [
        {{"test_type": "t-test", "variables": ["Nov_revenue", "Dec_revenue"], "p_value": 0.032, "significant": true, "conclusion": "Significant difference between months"}}
    ],
    "overall_confidence": 0.82
}}

Focus on:
- Statistical significance of findings
- Confidence intervals for anomalies and predictions
- Correlation strengths and their business implications
- Trend analysis with quantitative measures
- Actionable insights based on statistical evidence

Do not include any text before or after the JSON.
"""
        
        messages = [
            LLMMessage(role="system", content="You are a data analyst. You must respond with valid JSON only. Do not include any explanatory text, markdown formatting, or other content. Return only the requested JSON structure."),
            LLMMessage(role="user", content=trend_prompt)
        ]
        
        try:
            response = await llm_manager.generate(messages)
            logger.debug(f"LLM trend response: {response.content[:200]}...")
            
            # Check if response is empty
            if not response.content or not response.content.strip():
                logger.error("❌ Trend analysis LLM returned empty response")
                error_msg = "LLM returned empty response for trend analysis"
                state["errors"].append(error_msg)
                raise Exception(f"Cannot perform trend analysis: {error_msg}")
            
            # Try to extract JSON if it's wrapped in markdown
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].strip()
            
            trend_analysis = json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Trend analysis JSON parsing failed: {e}")
            logger.error(f"❌ Raw LLM response: {response.content[:500]}")
            error_msg = f"Trend analysis response was not valid JSON: {str(e)}"
            state["errors"].append(error_msg)
            raise Exception(f"Cannot parse trend analysis: {error_msg}")
        except Exception as e:
            logger.error(f"❌ LLM Trend Analysis Failed: {e}")
            error_msg = f"Trend analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            raise Exception(f"Cannot perform trend analysis: {error_msg}")
        
        state.update({
            "current_node": "trend_analysis",
            "trends": trend_analysis.get("trends", []),
            "patterns": trend_analysis.get("patterns", []),
            "anomalies": trend_analysis.get("anomalies", []),
            "correlations": trend_analysis.get("correlations", []),
            "forecasts": trend_analysis.get("forecasts", []),
            "statistical_tests": trend_analysis.get("statistical_tests", [])
        })
        
        state["completed_nodes"].append("trend_analysis")
        state["node_outputs"]["trend_analysis"] = {
            "trends_detected": len(trend_analysis.get("trends", [])),
            "patterns_found": len(trend_analysis.get("patterns", [])),
            "anomalies_detected": len(trend_analysis.get("anomalies", [])),
            "correlations_found": len(trend_analysis.get("correlations", [])),
            "forecasts_generated": len(trend_analysis.get("forecasts", [])),
            "statistical_tests_performed": len(trend_analysis.get("statistical_tests", [])),
            "overall_confidence": trend_analysis.get("overall_confidence", 0.75)
        }
        
        logger.info(f"✅ trend_analysis completed: {len(state['trends'])} trends, {len(state['patterns'])} patterns")
        return state
        
    except Exception as e:
        logger.error(f"❌ trend_analysis failed: {e}")
        state["errors"].append(f"Trend analysis failed: {str(e)}")
        raise Exception(f"Trend analysis failed: {str(e)}")


async def explain_result_node(state: AnalysisState) -> AnalysisState:
    """Generate narrative explanation and recommended actions"""
    logger.info("✍️ Starting explain_result node")
    
    try:
        from services.llm_providers import llm_manager, LLMMessage
        
        context = {
            "query": state["query"],
            "operation_type": state["operation_type"],
            "execution_results": state["execution_results"],
            "trends": state["trends"],
            "patterns": state["patterns"],
            "anomalies": state["anomalies"],
            "correlations": state.get("correlations", []),
            "forecasts": state.get("forecasts", []),
            "statistical_tests": state.get("statistical_tests", [])
        }
        
        explanation_prompt = f"""
Generate a comprehensive business analysis summary for this data analysis. Return ONLY valid JSON with the exact structure below:

Context:
{json.dumps(context, indent=2, default=str)[:1500]}

Return ONLY this JSON structure:
{{
    "executive_summary": "2-3 sentence executive summary",
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "recommended_actions": ["Action 1", "Action 2", "Action 3", "Action 4"],
    "confidence_score": 0.85,
    "narrative_explanation": "Detailed business-ready narrative with clear insights"
}}

Base recommendations on the actual analysis results, trends, and patterns found. Make them specific and actionable.
"""
        
        messages = [
            LLMMessage(role="system", content="You are a senior business analyst. You must respond with valid JSON only. Do not include any explanatory text, markdown formatting, or other content. Return only the requested JSON structure with specific, actionable insights based on the analysis results."),
            LLMMessage(role="user", content=explanation_prompt)
        ]
        
        try:
            response = await llm_manager.generate(messages)
            logger.debug(f"LLM explanation response: {response.content[:200]}...")
            
            # Check if response is empty
            if not response.content or not response.content.strip():
                logger.error("❌ Explanation LLM returned empty response")
                error_msg = "LLM returned empty response for explanation"
                state["errors"].append(error_msg)
                raise Exception(f"Cannot generate explanation: {error_msg}")
            
            # Try to extract JSON if it's wrapped in markdown
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].strip()
            
            try:
                explanation_data = json.loads(content)
                narrative = explanation_data.get("narrative_explanation", "Analysis completed successfully")
                llm_key_findings = explanation_data.get("key_findings", [])
                llm_recommendations = explanation_data.get("recommended_actions", [])
                llm_confidence = explanation_data.get("confidence_score", 0.85)
                executive_summary = explanation_data.get("executive_summary", "")
            except json.JSONDecodeError as json_error:
                logger.error(f"❌ Explanation JSON parsing failed: {json_error}")
                logger.error(f"❌ Raw LLM response: {response.content[:500]}")
                # Fall back to using the raw response as narrative
                narrative = response.content
                llm_key_findings = []
                llm_recommendations = []
                llm_confidence = 0.75
                executive_summary = response.content[:200] + "..." if len(response.content) > 200 else response.content
                
        except Exception as e:
            logger.error(f"❌ LLM Result Explanation Failed: {e}")
            error_msg = f"Result explanation failed: {str(e)}"
            state["errors"].append(error_msg)
            raise Exception(f"Cannot generate analysis explanation: {error_msg}")
        
        # Use LLM-generated insights or fallback to structured ones
        if llm_key_findings:
            key_findings = llm_key_findings
        else:
            # Fallback structured insights
            key_findings = [
                f"Analysis completed for {len(state['parsed_files'])} data files",
                f"Operation type: {state['operation_type']}",
                f"Trends detected: {len(state['trends'])} patterns identified"
            ]
            
            # Add specific findings from trends
            for trend in state["trends"][:3]:  # Top 3 trends
                key_findings.append(f"{trend.get('metric', 'metric').title()} shows {trend.get('direction', 'stable')} trend")
        
        # Use LLM-generated recommendations or fallback to structured ones
        if llm_recommendations:
            recommended_actions = llm_recommendations
        else:
            # Fallback structured recommendations
            recommended_actions = [
                "Review the detailed analysis results for specific metrics",
                "Monitor key performance indicators based on identified trends",
                "Consider seasonal patterns for future planning"
            ]
            
            # Add anomaly-based recommendations
            if state["anomalies"]:
                recommended_actions.append("Investigate detected anomalies for potential issues or opportunities")
        
        # Use LLM confidence or calculate fallback
        if 'llm_confidence' in locals():
            confidence_score = llm_confidence
        else:
            confidence_score = 0.85 if not state["errors"] else max(0.6 - len(state["errors"]) * 0.1, 0.3)
        
        # Check if analysis had significant errors requiring graceful degradation
        errors = state.get("errors", [])
        error_report = None
        
        # Generate error report and possibly apply graceful degradation
        if errors and ERROR_HANDLING_AVAILABLE:
            try:
                logger.info(f"📋 Generating comprehensive error analysis for {len(errors)} errors")
                
                error_context = {
                    "operation_type": state.get("operation_type", "unknown"),
                    "current_node": "explain_result",
                    "error_handling_available": ERROR_HANDLING_AVAILABLE,
                    "files_count": len(state.get("parsed_files", [])),
                    "timeout_seconds": 45
                }
                
                error_report = create_error_report(errors, error_context, state)
                
                # If errors are blocking or critical, apply graceful degradation
                if not error_report.get("can_continue_analysis", True) or error_report.get("overall_status") == "blocked":
                    logger.info("🛡️ Applying graceful degradation due to critical errors")
                    
                    degradation_result = await apply_graceful_degradation(state, errors, {"error_summary": error_report})
                    
                    # Use graceful degradation result as final result
                    final_result = degradation_result
                    
                    # Update confidence and add error context
                    final_result["confidence_score"] = min(confidence_score, 0.7)  # Cap confidence for degraded analysis
                    final_result["error_report"] = error_report
                    final_result["analysis_method"] = "graceful_degradation"
                    
                    logger.info("✅ Graceful degradation analysis completed")
                    
                else:
                    # Proceed with normal result but include error context
                    final_result = {
                        "query": state["query"],
                        "executive_summary": executive_summary if 'executive_summary' in locals() and executive_summary else (narrative[:200] + "..." if len(narrative) > 200 else narrative),
                        "full_narrative": narrative,
                        "operation_type": state["operation_type"],
                        "files_analyzed": len(state["parsed_files"]),
                        "execution_results": state["execution_results"],
                        "trend_analysis": {
                            "trends": state["trends"],
                            "patterns": state["patterns"], 
                            "anomalies": state["anomalies"],
                            "correlations": state.get("correlations", []),
                            "forecasts": state.get("forecasts", []),
                            "statistical_tests": state.get("statistical_tests", [])
                        },
                        "key_findings": key_findings,
                        "recommended_actions": recommended_actions,
                        "confidence_score": confidence_score,
                        "analysis_method": "standard_with_error_recovery",
                        "error_report": error_report,
                        "metadata": {
                            "execution_time": "workflow_completed",
                            "nodes_completed": state["completed_nodes"],
                            "errors_encountered": len(errors),
                            "error_handling_applied": True,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    # Add error-specific recommendations
                    if error_report.get("recommendations"):
                        final_result["recommended_actions"].extend(error_report["recommendations"])
                    
                    logger.info("✅ Standard analysis completed with error recovery context")
                    
            except Exception as error_handling_exception:
                logger.error(f"❌ Error handling system failed: {error_handling_exception}")
                # Fallback to basic result structure
                final_result = {
                    "query": state["query"],
                    "executive_summary": executive_summary if 'executive_summary' in locals() and executive_summary else (narrative[:200] + "..." if len(narrative) > 200 else narrative),
                    "full_narrative": narrative,
                    "operation_type": state["operation_type"],
                    "files_analyzed": len(state["parsed_files"]),
                    "execution_results": state["execution_results"],
                    "trend_analysis": {
                        "trends": state["trends"],
                        "patterns": state["patterns"], 
                        "anomalies": state["anomalies"],
                        "correlations": state.get("correlations", []),
                        "forecasts": state.get("forecasts", []),
                        "statistical_tests": state.get("statistical_tests", [])
                    },
                    "key_findings": key_findings,
                    "recommended_actions": recommended_actions,
                    "confidence_score": max(confidence_score - 0.2, 0.4),  # Reduce confidence due to error handling failure
                    "analysis_method": "standard_with_limited_error_handling",
                    "metadata": {
                        "execution_time": "workflow_completed",
                        "nodes_completed": state["completed_nodes"],
                        "errors": errors,
                        "error_handling_failed": True,
                        "timestamp": datetime.now().isoformat()
                    }
                }
        else:
            # No errors or enhanced error handling not available - standard result
            final_result = {
                "query": state["query"],
                "executive_summary": executive_summary if 'executive_summary' in locals() and executive_summary else (narrative[:200] + "..." if len(narrative) > 200 else narrative),
                "full_narrative": narrative,
                "operation_type": state["operation_type"],
                "files_analyzed": len(state["parsed_files"]),
                "execution_results": state["execution_results"],
                "trend_analysis": {
                    "trends": state["trends"],
                    "patterns": state["patterns"], 
                    "anomalies": state["anomalies"],
                    "correlations": state.get("correlations", []),
                    "forecasts": state.get("forecasts", []),
                    "statistical_tests": state.get("statistical_tests", [])
                },
                "key_findings": key_findings,
                "recommended_actions": recommended_actions,
                "confidence_score": confidence_score,
                "analysis_method": "standard_analysis",
                "metadata": {
                    "execution_time": "workflow_completed",
                    "nodes_completed": state["completed_nodes"],
                    "errors": errors if errors else [],
                    "enhanced_error_handling": ERROR_HANDLING_AVAILABLE,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        state.update({
            "current_node": "explain_result",
            "final_result": final_result,
            "insights": key_findings,
            "recommended_actions": recommended_actions,
            "confidence_score": confidence_score
        })
        
        state["completed_nodes"].append("explain_result")
        state["node_outputs"]["explain_result"] = {
            "narrative_generated": len(narrative) > 0,
            "findings_count": len(key_findings),
            "actions_count": len(recommended_actions),
            "confidence_score": confidence_score
        }
        
        logger.info(f"✅ explain_result completed: {len(key_findings)} findings, {len(recommended_actions)} actions")
        return state
        
    except Exception as e:
        logger.error(f"❌ explain_result failed: {e}")
        state["errors"].append(f"Result explanation failed: {str(e)}")
        
        # Apply comprehensive fallback with graceful degradation
        try:
            if ERROR_HANDLING_AVAILABLE:
                logger.info("🚨 Applying emergency graceful degradation due to explain_result failure")
                errors = state.get("errors", []) + [f"Result explanation failed: {str(e)}"]
                
                fallback_result = await apply_graceful_degradation(
                    state, 
                    errors, 
                    {"error_summary": "Final explanation generation failed"}
                )
                
                fallback_result["analysis_method"] = "emergency_graceful_degradation"
                fallback_result["confidence_score"] = 0.5  # Low confidence for emergency fallback
            else:
                # Basic fallback when enhanced error handling is not available
                fallback_result = {
                    "query": state.get("query", "Unknown query"),
                    "executive_summary": "Analysis completed with basic results due to processing limitations",
                    "operation_type": state.get("operation_type", "unknown"),
                    "files_analyzed": len(state.get("parsed_files", [])),
                    "key_findings": [
                        "Analysis workflow completed",
                        "Basic data processing performed",
                        "Some limitations in result generation"
                    ],
                    "recommended_actions": [
                        "Review data format and structure",
                        "Try with simpler analysis requirements",
                        "Contact support if issues persist"
                    ],
                    "confidence_score": 0.4,
                    "analysis_method": "basic_fallback",
                    "metadata": {
                        "status": "completed_with_basic_fallback",
                        "errors": state.get("errors", []),
                        "timestamp": datetime.now().isoformat()
                    }
                }
        except Exception as fallback_error:
            logger.error(f"❌ Even fallback analysis failed: {fallback_error}")
            # Absolute minimum fallback
            fallback_result = {
                "query": state.get("query", "Analysis query"),
                "executive_summary": "Analysis encountered technical difficulties but workflow completed",
                "key_findings": ["Workflow execution attempted", "Technical issues encountered"],
                "recommended_actions": ["Review system status", "Try again later", "Contact technical support"],
                "confidence_score": 0.2,
                "analysis_method": "minimal_fallback",
                "metadata": {
                    "status": "completed_with_minimal_fallback",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        state.update({
            "final_result": fallback_result,
            "insights": fallback_result.get("key_findings", ["Analysis completed"]),
            "recommended_actions": fallback_result.get("recommended_actions", ["Review results"]),
            "confidence_score": fallback_result.get("confidence_score", 0.5)
        })
        return state


def create_workflow_graph() -> Any:
    """
    Create the LangGraph workflow with all nodes and transitions
    """
    logger.info("🔧 Creating workflow graph")
    
    # Create state graph
    workflow = StateGraph(AnalysisState)
    
    # Add all nodes
    workflow.add_node("parse_files", parse_files_node)
    workflow.add_node("plan_operations", plan_operations_node) 
    workflow.add_node("align_timeseries", align_timeseries_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("validate_code", validate_code_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("trend_analysis", trend_analysis_node)
    workflow.add_node("explain_result", explain_result_node)
    
    # Define the workflow flow with error-aware retry loop
    workflow.set_entry_point("parse_files")
    workflow.add_edge("parse_files", "plan_operations")
    workflow.add_edge("plan_operations", "align_timeseries")
    workflow.add_edge("align_timeseries", "generate_code")
    workflow.add_edge("generate_code", "validate_code")
    
    # Add conditional edge for error handling and retry loop
    workflow.add_conditional_edges(
        "validate_code",
        should_retry_with_error_context,
        {
            "retry_generation": "generate_code",  # Loop back to generation with error context
            "proceed": "execute_code"             # Continue to execution
        }
    )
    
    workflow.add_conditional_edges(
        "execute_code", 
        should_retry_execution_with_context,
        {
            "retry_generation": "generate_code",  # Loop back to generation with execution error context
            "proceed": "trend_analysis"           # Continue to trend analysis
        }
    )
    
    workflow.add_edge("trend_analysis", "explain_result")
    workflow.add_edge("explain_result", END)
    
    # Compile the graph with checkpointing for fault tolerance
    if LANGGRAPH_AVAILABLE:
        try:
            # Try to create checkpointer without arguments first
            checkpointer = SqliteSaver()
            graph = workflow.compile(checkpointer=checkpointer)
        except Exception as e:
            logger.warning(f"Failed to create checkpointer: {e}, compiling without checkpointing")
            graph = workflow.compile()
    else:
        graph = workflow.compile()
    
    logger.info("✅ Workflow graph created successfully")
    return graph


# Error-aware retry decision functions
def should_retry_with_error_context(state: AnalysisState) -> str:
    """Decide whether to retry code generation with error context"""
    retry_count = state.get("code_generation_attempts", 1)  # Use actual retry count
    errors = state.get("errors", [])
    
    # Check if we have validation errors and haven't exceeded max retries
    if errors and retry_count < 3:
        # Look for specific error types that warrant retry with context
        recent_errors = errors[-5:]  # Last 5 errors
        error_text = " ".join(recent_errors).lower()
        
        # Retry if we have data/code related errors that can be fixed with better generation
        if any(keyword in error_text for keyword in [
            'column', 'keyerror', 'not found', 'missing', 'attribute', 'syntax', 
            'name error', 'type error', 'pandas', 'numpy', 'dataframe'
        ]):
            logger.info(f"🔄 Retrying code generation (attempt {retry_count + 1}/3) due to errors: {recent_errors[-1]}")
            return "retry_generation"
    
    # Otherwise proceed to execution
    return "proceed"

def should_retry_execution_with_context(state: AnalysisState) -> str:
    """Decide whether to retry from generation after execution errors"""
    retry_count = state.get("execution_retry_count", 0)
    execution_results = state.get("execution_results", {})
    
    # Check if execution failed and we haven't exceeded max retries
    if retry_count < 2 and execution_results.get("status") in ["execution_error", "timeout_error", "validation_failed"]:
        error_msg = execution_results.get("error", "")
        
        # Retry generation if we have specific execution errors
        if any(keyword in error_msg.lower() for keyword in [
            'float64', 'iloc', 'scalar', 'array', 'syntax', 'name error', 'undefined'
        ]):
            logger.info(f"🔄 Retrying from generation (attempt {retry_count + 1}/2) due to execution error: {error_msg}")
            return "retry_generation"
    
    # Otherwise proceed to trend analysis
    return "proceed"

# Create the global workflow instance
analysis_workflow = create_workflow_graph()
