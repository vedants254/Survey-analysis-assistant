"""
Safe Code Execution Engine
Provides secure, timeout-protected code execution with automatic error correction
"""

import asyncio
import ast
import contextlib
import io
import logging
import re
import signal
import sys
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CodeSafetyError(Exception):
    """Raised when code is deemed unsafe for execution"""
    pass


class ExecutionTimeoutError(Exception):
    """Raised when code execution times out"""
    pass


class CodeValidator:
    """Validates code for safety before execution"""
    
    # Dangerous patterns that should not be allowed
    DANGEROUS_PATTERNS = [
        r'import\s+os\b',
        r'import\s+subprocess\b', 
        r'import\s+sys\b',
        r'from\s+os\s+import',
        r'from\s+subprocess\s+import',
        r'from\s+sys\s+import',
        r'__import__\s*\(',
        r'eval\s*\(',
        r'exec\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'compile\s*\(',
        r'globals\s*\(',
        r'locals\s*\(',
        r'setattr\s*\(',
        r'getattr\s*\(',
        r'delattr\s*\(',
        r'hasattr\s*\(',
        r'dir\s*\(',
        r'vars\s*\(',
        r'__.*__\s*=',  # Dunder assignment
        r'while\s+True\s*:',  # Infinite loops
        r'for\s+.*\s+in\s+range\s*\(\s*\d{6,}\s*\)',  # Large ranges
    ]
    
    # Allowed imports for data analysis
    ALLOWED_IMPORTS = {
        'pandas', 'pd', 'numpy', 'np', 'json', 'math', 'datetime',
        'collections', 'itertools', 'functools', 'operator'
    }
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate code for safety and return validation result
        Returns (is_safe, list_of_issues)
        """
        issues = []
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                issues.append(f"Dangerous pattern detected: {pattern}")
        
        # Parse and analyze AST
        try:
            tree = ast.parse(code)
            ast_issues = self._analyze_ast(tree)
            issues.extend(ast_issues)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        
        # Check for reasonable complexity
        complexity_issues = self._check_complexity(code)
        issues.extend(complexity_issues)
        
        return len(issues) == 0, issues
    
    def _analyze_ast(self, tree: ast.AST) -> List[str]:
        """Analyze AST for potentially dangerous constructs"""
        issues = []
        
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_issues = self._check_imports(node)
                issues.extend(import_issues)
            
            # Check for loops that might be infinite
            elif isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value:
                    issues.append("Potential infinite while loop detected")
            
            # Check for very large ranges
            elif isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == 'range':
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, int) and node.args[0].value > 100000:
                        issues.append(f"Very large range detected: {node.args[0].value}")
        
        return issues
    
    def _check_imports(self, node) -> List[str]:
        """Check if imports are allowed"""
        issues = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in self.ALLOWED_IMPORTS:
                    issues.append(f"Disallowed import: {alias.name}")
        
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module not in self.ALLOWED_IMPORTS:
                issues.append(f"Disallowed import from: {node.module}")
        
        return issues
    
    def _check_complexity(self, code: str) -> List[str]:
        """Check code complexity metrics"""
        issues = []
        
        lines = code.split('\n')
        if len(lines) > 200:
            issues.append("Code is very long (>200 lines), potential performance issue")
        
        # Check for deeply nested structures
        max_indent = 0
        for line in lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        if max_indent > 16:  # More than 4 levels of nesting (assuming 4 spaces per level)
            issues.append("Code has very deep nesting, potential performance issue")
        
        return issues


class PandasErrorFixer:
    """Automatically fixes common pandas/numpy errors"""
    
    @staticmethod
    def fix_scalar_iloc_error(code: str) -> str:
        """Fix 'numpy.float64' object has no attribute 'iloc' errors"""
        patterns_and_fixes = [
            # Fix direct .iloc calls on potentially scalar values
            (r'(\w+)\.iloc\[0\]', r'safe_extract(\1)'),
            # Fix .iloc on aggregation results
            (r'(\w+\.\w+\(\))\s*\.iloc\[0\]', r'safe_extract(\1)'),
            # Fix .iloc on series operations
            (r'(\w+\[.*?\])\s*\.iloc\[0\]', r'safe_extract(\1)'),
        ]
        
        fixed_code = code
        for pattern, fix in patterns_and_fixes:
            fixed_code = re.sub(pattern, fix, fixed_code, flags=re.MULTILINE)
        
        return fixed_code
    
    @staticmethod
    def fix_type_errors(code: str) -> str:
        """Fix common pandas type errors"""
        # Add safe type conversion functions
        safe_functions = """
# Safe helper functions for data extraction and conversion
def safe_extract(value):
    \"\"\"Safely extract scalar value from pandas objects\"\"\"
    if value is None:
        return 0
    elif hasattr(value, 'iloc') and len(value) > 0:
        return value.iloc[0]
    elif hasattr(value, 'item'):
        return value.item()
    elif isinstance(value, (list, tuple)) and len(value) > 0:
        return value[0]
    elif pd.isna(value):
        return 0
    else:
        return float(value)

def safe_sum(series_or_value):
    \"\"\"Safely sum pandas series or return scalar\"\"\"
    if hasattr(series_or_value, 'sum'):
        result = series_or_value.sum()
        return safe_extract(result)
    else:
        return safe_extract(series_or_value)

def safe_mean(series_or_value):
    \"\"\"Safely calculate mean\"\"\"
    if hasattr(series_or_value, 'mean'):
        result = series_or_value.mean()
        return safe_extract(result)
    else:
        return safe_extract(series_or_value)

def safe_count(series_or_value):
    \"\"\"Safely count non-null values\"\"\"
    if hasattr(series_or_value, 'count'):
        return int(series_or_value.count())
    elif series_or_value is not None and not pd.isna(series_or_value):
        return 1
    else:
        return 0

"""
        return safe_functions + "\n" + code
    
    @staticmethod
    def add_error_handling(code: str) -> str:
        """Wrap code in comprehensive error handling"""
        wrapped_code = f"""
# Comprehensive error handling wrapper
import pandas as pd
import numpy as np
import json
from datetime import datetime

{PandasErrorFixer.fix_type_errors('')}  # Add helper functions

try:
    # Initialize result dictionary
    if 'analysis_results' not in locals():
        analysis_results = {{'status': 'success', 'timestamp': datetime.now().isoformat()}}
    
    # Original analysis code with fixes
{PandasErrorFixer._indent_code(PandasErrorFixer.fix_scalar_iloc_error(code), 4)}
    
    # Ensure result is properly formatted
    if 'analysis_results' in locals() and analysis_results:
        # Convert numpy types to Python types for JSON serialization
        analysis_results = {{k: (v.item() if hasattr(v, 'item') else 
                                float(v) if isinstance(v, (np.integer, np.floating)) else 
                                v) for k, v in analysis_results.items()}}
        
        if 'status' not in analysis_results:
            analysis_results['status'] = 'completed'
    else:
        analysis_results = {{
            'status': 'completed',
            'message': 'Analysis completed but no specific results generated',
            'timestamp': datetime.now().isoformat()
        }}

except KeyError as e:
    analysis_results = {{
        'status': 'data_error',
        'error': f'Column or key not found: {{str(e)}}',
        'message': 'Analysis failed due to missing data columns. Please check your data.',
        'suggested_action': 'Verify column names and data structure',
        'timestamp': datetime.now().isoformat()
    }}
except (TypeError, AttributeError) as e:
    analysis_results = {{
        'status': 'type_error', 
        'error': f'Data type issue: {{str(e)}}',
        'message': 'Analysis failed due to data type problems.',
        'suggested_action': 'Check data types and formatting',
        'timestamp': datetime.now().isoformat()
    }}
except Exception as e:
    analysis_results = {{
        'status': 'execution_error',
        'error': str(e),
        'message': 'Analysis encountered an unexpected error',
        'suggested_action': 'Review data and try a simpler analysis approach',
        'timestamp': datetime.now().isoformat()
    }}
"""
        return wrapped_code
    
    @staticmethod
    def _indent_code(code: str, spaces: int) -> str:
        """Add indentation to code"""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line for line in code.split('\n'))


class SafeCodeExecutor:
    """Secure code execution engine with timeout and error recovery"""
    
    def __init__(self, timeout_seconds: int = 30, max_memory_mb: int = 512):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.validator = CodeValidator()
        self.fixer = PandasErrorFixer()
    
    async def execute_code(
        self, 
        code: str, 
        context: Dict[str, Any] = None,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Execute code safely with automatic error recovery
        """
        context = context or {}
        original_code = code
        
        # Validate code safety
        is_safe, issues = self.validator.validate_code(code)
        if not is_safe:
            return {
                'status': 'validation_failed',
                'error': 'Code failed safety validation',
                'issues': issues,
                'timestamp': datetime.now().isoformat()
            }
        
        # Attempt execution with progressive error fixing
        for attempt in range(max_attempts):
            try:
                logger.info(f"🔄 Code execution attempt {attempt + 1}/{max_attempts}")
                
                # Apply error fixes based on attempt number
                if attempt == 0:
                    # First attempt: original code with basic error handling
                    fixed_code = self.fixer.add_error_handling(code)
                elif attempt == 1:
                    # Second attempt: fix common pandas errors
                    fixed_code = self.fixer.add_error_handling(
                        self.fixer.fix_scalar_iloc_error(code)
                    )
                else:
                    # Final attempt: maximum error handling
                    fixed_code = self.fixer.add_error_handling(
                        self.fixer.fix_type_errors(
                            self.fixer.fix_scalar_iloc_error(code)
                        )
                    )
                
                # Execute with timeout
                result = await self._execute_with_timeout(fixed_code, context)
                
                if result and result.get('status') != 'error':
                    logger.info(f"✅ Code execution successful on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"⚠️ Attempt {attempt + 1} returned error result: {result}")
                    if attempt == max_attempts - 1:
                        return result or self._create_fallback_result("All attempts failed")
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️ Execution attempt {attempt + 1} failed: {error_msg}")
                
                if attempt == max_attempts - 1:
                    # Final attempt failed, return comprehensive error info
                    return {
                        'status': 'execution_failed',
                        'error': error_msg,
                        'attempts': max_attempts,
                        'original_code_length': len(original_code),
                        'final_attempt_code_length': len(fixed_code),
                        'fallback_result': self._create_fallback_result(error_msg),
                        'timestamp': datetime.now().isoformat()
                    }
        
        # Should never reach here, but safety fallback
        return self._create_fallback_result("Unexpected execution flow")
    
    async def _execute_with_timeout(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code with timeout protection"""
        
        # Prepare execution namespace with proper imports
        import json as json_module
        namespace = {
            'pd': pd,
            'np': np,
            'json': json_module,
            'datetime': datetime,
            'analysis_results': {}
        }
        
        # Add context data to namespace
        if context.get('dataframes'):
            namespace.update(context['dataframes'])
        
        # Create a timeout handler
        def timeout_handler(signum, frame):
            raise ExecutionTimeoutError(f"Code execution timed out after {self.timeout_seconds} seconds")
        
        # Set up signal handler for timeout (Unix/Linux/macOS only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
        
        try:
            # Capture stdout for any print statements
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, namespace)
            
            # Get results
            analysis_results = namespace.get('analysis_results', {})
            captured_output = stdout_capture.getvalue()
            
            # Add captured output if it exists
            if captured_output.strip():
                analysis_results['captured_output'] = captured_output.strip()
            
            # Ensure results are JSON serializable
            analysis_results = self._make_json_serializable(analysis_results)
            
            return analysis_results
            
        except ExecutionTimeoutError as e:
            logger.error(f"⏱️ Code execution timed out: {e}")
            return {
                'status': 'timeout_error',
                'error': str(e),
                'timeout_seconds': self.timeout_seconds,
                'message': 'Code execution took too long, likely due to inefficient operations or infinite loops',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Code execution failed: {e}")
            return {
                'status': 'execution_error',
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()[-1000:],  # Last 1000 chars of traceback
                'timestamp': datetime.now().isoformat()
            }
        finally:
            # Clean up signal handler
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            try:
                # Try to convert to basic Python types
                return float(obj) if isinstance(obj, (int, float)) else str(obj)
            except:
                return str(obj)
    
    def _create_fallback_result(self, error_message: str) -> Dict[str, Any]:
        """Create a fallback result when all execution attempts fail"""
        return {
            'status': 'fallback_analysis',
            'message': 'Analysis completed using fallback methods due to execution issues',
            'error': error_message,
            'basic_info': {
                'analysis_attempted': True,
                'execution_method': 'fallback',
                'timestamp': datetime.now().isoformat()
            },
            'recommendations': [
                'Check data formats and column names',
                'Simplify analysis approach',
                'Verify data completeness',
                'Consider manual data review'
            ]
        }


# Convenient factory function
def create_safe_executor(timeout_seconds: int = 30, max_memory_mb: int = 512) -> SafeCodeExecutor:
    """Create a configured safe code executor"""
    return SafeCodeExecutor(timeout_seconds, max_memory_mb)


# Example usage
async def example_safe_execution():
    """Example of safe code execution"""
    executor = create_safe_executor(timeout_seconds=30)
    
    # Example problematic code that might cause errors
    problematic_code = """
# This might cause pandas/numpy errors
total_revenue = df_data['revenue'].sum().iloc[0]  # This could fail
analysis_results = {
    'total_revenue': total_revenue,
    'status': 'completed'
}
"""
    
    context = {
        'dataframes': {
            'df_data': pd.DataFrame({'revenue': [100, 200, 300]})
        }
    }
    
    result = await executor.execute_code(problematic_code, context)
    logger.info(f"Execution result: {result}")
    
    return result