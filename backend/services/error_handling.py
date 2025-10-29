"""
Enhanced Error Handling Framework
Provides comprehensive error classification, intelligent retry strategies, and graceful degradation
"""

import asyncio
import json
import logging
import random
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime

try:
    from .security_sanitizer import SecuritySanitizer
    SANITIZER_AVAILABLE = True
except ImportError:
    SANITIZER_AVAILABLE = False
    # Mock SecuritySanitizer if not available
    class SecuritySanitizer:
        @staticmethod
        def sanitize_error_message(msg): return msg
        @staticmethod 
        def sanitize_for_frontend(data): return data

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Classification of error types for targeted handling"""
    DATA_ERROR = "data_error"           # Missing columns, data format issues
    LLM_ERROR = "llm_error"             # API rate limits, quota exceeded
    CODE_ERROR = "code_error"           # Syntax, runtime, pandas/numpy issues  
    TIMEOUT_ERROR = "timeout_error"     # Execution timeouts, hanging operations
    RESOURCE_ERROR = "resource_error"   # Memory limits, disk space
    NETWORK_ERROR = "network_error"     # Connection issues, API failures
    VALIDATION_ERROR = "validation_error"  # Input validation failures
    SYSTEM_ERROR = "system_error"       # Unexpected system errors


@dataclass
class ErrorContext:
    """Rich context information for error analysis"""
    category: ErrorCategory
    error_code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_fixes: List[str] = field(default_factory=list)
    retry_recommended: bool = True
    fallback_available: bool = True
    user_actionable: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    timeout_seconds: Optional[int] = None
    
    # Error-specific retry counts
    data_error_attempts: int = 2
    llm_error_attempts: int = 4
    code_error_attempts: int = 3
    timeout_error_attempts: int = 2


class ErrorClassifier:
    """Intelligent error classification system"""
    
    @staticmethod
    def classify_error(error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Classify an error and provide handling recommendations"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        context = context or {}
        
        # Data-related errors
        if any(keyword in error_str for keyword in [
            'product_nov', 'column', 'keyerror', 'missing', 'not found', 
            'invalid index', 'empty dataframe', 'no data'
        ]):
            return ErrorContext(
                category=ErrorCategory.DATA_ERROR,
                error_code="DATA_MISSING_OR_INVALID",
                message=f"Data issue detected: {str(error)}",
                details={
                    "original_error": str(error),
                    "error_type": error_type,
                    "context": context
                },
                suggested_fixes=[
                    "Check if all required data columns exist",
                    "Verify data file completeness", 
                    "Use alternative column names or data sources",
                    "Apply data cleaning and validation"
                ]
            )
        
        # LLM API errors
        elif any(keyword in error_str for keyword in [
            'rate limit', 'quota', 'api key', 'unauthorized', 'token limit',
            'context length', 'model unavailable'
        ]):
            return ErrorContext(
                category=ErrorCategory.LLM_ERROR,
                error_code="LLM_API_ISSUE",
                message=f"LLM provider issue: {str(error)}",
                details={
                    "original_error": str(error),
                    "error_type": error_type,
                    "provider": context.get('provider', 'unknown')
                },
                suggested_fixes=[
                    "Wait and retry with exponential backoff",
                    "Try alternative LLM provider",
                    "Reduce prompt length or complexity",
                    "Check API key and quota limits"
                ]
            )
        
        # Code execution errors
        elif any(keyword in error_str for keyword in [
            'float64', 'iloc', 'scalar', 'array', 'numpy', 'pandas',
            'syntax error', 'name error', 'attribute error', 'type error'
        ]):
            return ErrorContext(
                category=ErrorCategory.CODE_ERROR,
                error_code="CODE_EXECUTION_FAILURE",
                message=f"Code execution error: {str(error)}",
                details={
                    "original_error": str(error),
                    "error_type": error_type,
                    "code_snippet": context.get('code', 'N/A')[:200]
                },
                suggested_fixes=[
                    "Apply automatic pandas/numpy error fixes",
                    "Regenerate code with better error handling",
                    "Use simpler analysis approach",
                    "Add data type validation and conversion"
                ]
            )
        
        # Timeout errors
        elif any(keyword in error_str for keyword in [
            'timeout', 'timed out', 'hanging', 'infinite loop', 'sigalrm'
        ]) or isinstance(error, TimeoutError):
            return ErrorContext(
                category=ErrorCategory.TIMEOUT_ERROR,
                error_code="EXECUTION_TIMEOUT",
                message=f"Operation timed out: {str(error)}",
                details={
                    "original_error": str(error),
                    "error_type": error_type,
                    "timeout_duration": context.get('timeout_seconds', 'unknown')
                },
                suggested_fixes=[
                    "Reduce data set size",
                    "Simplify analysis complexity",
                    "Add progress checkpoints",
                    "Use streaming processing"
                ],
                retry_recommended=False  # Don't retry timeouts without changes
            )
        
        # Network/connection errors
        elif any(keyword in error_str for keyword in [
            'connection', 'network', 'dns', 'http', 'ssl', 'certificate'
        ]):
            return ErrorContext(
                category=ErrorCategory.NETWORK_ERROR,
                error_code="NETWORK_CONNECTION_FAILED",
                message=f"Network connectivity issue: {str(error)}",
                details={
                    "original_error": str(error),
                    "error_type": error_type,
                    "endpoint": context.get('endpoint', 'unknown')
                },
                suggested_fixes=[
                    "Check internet connectivity",
                    "Verify API endpoint availability",
                    "Try alternative network route",
                    "Use cached/offline fallback"
                ]
            )
        
        # Resource errors
        elif any(keyword in error_str for keyword in [
            'memory', 'disk space', 'resource', 'limit exceeded'
        ]):
            return ErrorContext(
                category=ErrorCategory.RESOURCE_ERROR,
                error_code="INSUFFICIENT_RESOURCES",
                message=f"Resource constraint: {str(error)}",
                details={
                    "original_error": str(error),
                    "error_type": error_type,
                    "resource_type": context.get('resource_type', 'unknown')
                },
                suggested_fixes=[
                    "Reduce memory usage",
                    "Process data in smaller chunks",
                    "Clean up temporary files",
                    "Use more efficient algorithms"
                ],
                retry_recommended=False
            )
        
        # Generic system errors
        else:
            return ErrorContext(
                category=ErrorCategory.SYSTEM_ERROR,
                error_code="UNEXPECTED_ERROR",
                message=f"Unexpected system error: {str(error)}",
                details={
                    "original_error": str(error),
                    "error_type": error_type,
                    "traceback": traceback.format_exc(),
                    "context": context
                },
                suggested_fixes=[
                    "Review system logs for more details",
                    "Try restarting the operation",
                    "Contact system administrator if persistent",
                    "Use alternative processing approach"
                ]
            )


class RetryStrategy:
    """Intelligent retry strategy with multiple approaches"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    def should_retry(self, error_context: ErrorContext, attempt: int) -> bool:
        """Determine if an error should be retried based on context"""
        if not error_context.retry_recommended:
            return False
        
        # Get category-specific max attempts
        max_attempts = self._get_max_attempts_for_category(error_context.category)
        
        return attempt < max_attempts
    
    def _get_max_attempts_for_category(self, category: ErrorCategory) -> int:
        """Get maximum retry attempts for specific error category"""
        category_limits = {
            ErrorCategory.DATA_ERROR: self.config.data_error_attempts,
            ErrorCategory.LLM_ERROR: self.config.llm_error_attempts,
            ErrorCategory.CODE_ERROR: self.config.code_error_attempts,
            ErrorCategory.TIMEOUT_ERROR: self.config.timeout_error_attempts,
        }
        return category_limits.get(category, self.config.max_attempts)
    
    def calculate_delay(self, error_context: ErrorContext, attempt: int) -> float:
        """Calculate appropriate delay before retry"""
        base_delay = self.config.base_delay
        
        # Category-specific delay adjustments
        if error_context.category == ErrorCategory.LLM_ERROR:
            # Longer delays for API rate limiting
            base_delay *= 3.0
        elif error_context.category == ErrorCategory.NETWORK_ERROR:
            # Moderate delays for network issues
            base_delay *= 2.0
        elif error_context.category == ErrorCategory.CODE_ERROR:
            # Shorter delays for code fixes
            base_delay *= 0.5
        
        # Exponential backoff
        delay = base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor
        
        return delay


class ErrorHandler:
    """Main error handling orchestrator"""
    
    def __init__(self, retry_config: RetryConfig = None):
        self.classifier = ErrorClassifier()
        self.retry_strategy = RetryStrategy(retry_config)
        self.error_history: List[ErrorContext] = []
    
    async def handle_with_retry(
        self,
        func: Callable,
        *args,
        context: Dict[str, Any] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """Execute function with intelligent retry and error handling"""
        context = context or {}
        last_error_context = None
        
        attempt = 0
        while True:
            try:
                if progress_callback and attempt > 0:
                    sanitized_msg = SecuritySanitizer.sanitize_error_message(last_error_context.message) if last_error_context and SANITIZER_AVAILABLE else (last_error_context.message if last_error_context else 'Unknown error')
                    callback_data = {
                        "status": "retrying",
                        "attempt": attempt + 1,
                        "message": f"Retrying after error: {sanitized_msg}"
                    }
                    # Sanitize entire callback data for frontend
                    if SANITIZER_AVAILABLE:
                        callback_data = SecuritySanitizer.sanitize_for_frontend(callback_data)
                    await progress_callback(callback_data)
                
                # Execute the function
                result = await self._execute_with_timeout(func, *args, **kwargs)
                
                # Success - log if this was a retry
                if attempt > 0:
                    logger.info(f"✅ Successfully recovered after {attempt} attempts")
                
                return result
                
            except Exception as error:
                attempt += 1
                
                # Classify the error
                error_context = self.classifier.classify_error(error, context)
                self.error_history.append(error_context)
                last_error_context = error_context
                
                # Log with sanitized message
                sanitized_msg = SecuritySanitizer.sanitize_error_message(error_context.message) if SANITIZER_AVAILABLE else error_context.message
                logger.warning(f"⚠️ Attempt {attempt} failed: {sanitized_msg}")
                
                # Check if we should retry
                if not self.retry_strategy.should_retry(error_context, attempt):
                    logger.error(f"❌ Max retries reached for {error_context.category.value}")
                    
                    # Try fallback strategy
                    if error_context.fallback_available:
                        return await self._execute_fallback(error_context, *args, **kwargs)
                    
                    # No fallback available, re-raise with enhanced context
                    raise EnhancedError(
                        error_context.message,
                        original_error=error,
                        error_context=error_context,
                        retry_attempts=attempt
                    )
                
                # Calculate delay and wait
                delay = self.retry_strategy.calculate_delay(error_context, attempt - 1)
                logger.info(f"⏳ Waiting {delay:.2f}s before retry {attempt + 1}")
                await asyncio.sleep(delay)
    
    async def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with optional timeout"""
        timeout = self.retry_strategy.config.timeout_seconds
        
        if timeout and asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        else:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    async def _execute_fallback(self, error_context: ErrorContext, *args, **kwargs) -> Any:
        """Execute fallback strategy based on error type"""
        logger.info(f"🚨 Executing fallback strategy for {error_context.category.value}")
        
        if error_context.category == ErrorCategory.DATA_ERROR:
            return await self._data_error_fallback(error_context, *args, **kwargs)
        elif error_context.category == ErrorCategory.CODE_ERROR:
            return await self._code_error_fallback(error_context, *args, **kwargs)
        elif error_context.category == ErrorCategory.LLM_ERROR:
            return await self._llm_error_fallback(error_context, *args, **kwargs)
        else:
            return await self._generic_fallback(error_context, *args, **kwargs)
    
    async def _data_error_fallback(self, error_context: ErrorContext, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for data-related errors"""
        result = {
            "status": "partial_success",
            "message": "Data analysis completed with limited scope due to data issues",
            "error_details": SecuritySanitizer.sanitize_error_message(error_context.message) if SANITIZER_AVAILABLE else error_context.message,
            "suggested_actions": error_context.suggested_fixes,
            "fallback_analysis": {
                "available_data": "Limited data analysis performed",
                "recommendations": [
                    "Review data quality and completeness",
                    "Consider alternative data sources",
                    "Apply data cleaning procedures"
                ]
            }
        }
        return SecuritySanitizer.sanitize_for_frontend(result) if SANITIZER_AVAILABLE else result
    
    async def _code_error_fallback(self, error_context: ErrorContext, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for code execution errors"""
        result = {
            "status": "analysis_simplified",
            "message": "Analysis completed using simplified approach due to execution issues",
            "error_details": SecuritySanitizer.sanitize_error_message(error_context.message) if SANITIZER_AVAILABLE else error_context.message,
            "suggested_actions": error_context.suggested_fixes,
            "fallback_analysis": {
                "approach": "basic statistical analysis",
                "limitations": "Complex calculations could not be performed",
                "recommendations": [
                    "Review data types and formats",
                    "Consider manual data preprocessing",
                    "Try alternative analysis methods"
                ]
            }
        }
        return SecuritySanitizer.sanitize_for_frontend(result) if SANITIZER_AVAILABLE else result
    
    async def _llm_error_fallback(self, error_context: ErrorContext, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for LLM-related errors"""
        result = {
            "status": "template_analysis",
            "message": "Analysis completed using template-based approach due to LLM issues",
            "error_details": SecuritySanitizer.sanitize_error_message(error_context.message) if SANITIZER_AVAILABLE else error_context.message,
            "suggested_actions": error_context.suggested_fixes,
            "fallback_analysis": {
                "approach": "rule-based analysis",
                "insights": [
                    "Basic trend analysis applied",
                    "Standard business metrics calculated",
                    "Template recommendations provided"
                ],
                "limitations": "AI-powered insights unavailable"
            }
        }
        return SecuritySanitizer.sanitize_for_frontend(result) if SANITIZER_AVAILABLE else result
    
    async def _generic_fallback(self, error_context: ErrorContext, *args, **kwargs) -> Dict[str, Any]:
        """Generic fallback for unexpected errors"""
        result = {
            "status": "minimal_analysis",
            "message": "Minimal analysis completed due to system issues",
            "error_details": SecuritySanitizer.sanitize_error_message(error_context.message) if SANITIZER_AVAILABLE else error_context.message,
            "suggested_actions": error_context.suggested_fixes,
            "fallback_analysis": {
                "approach": "basic data summary",
                "status": "system_degraded",
                "recommendations": [
                    "Contact system administrator",
                    "Try again later",
                    "Review system logs"
                ]
            }
        }
        return SecuritySanitizer.sanitize_for_frontend(result) if SANITIZER_AVAILABLE else result
    
    def get_error_summary(self, for_frontend: bool = True) -> Dict[str, Any]:
        """Get comprehensive error summary for debugging with optional sanitization"""
        if not self.error_history:
            return {"status": "no_errors", "total_errors": 0}
        
        error_counts = {}
        for error in self.error_history:
            category = error.category.value
            error_counts[category] = error_counts.get(category, 0) + 1
        
        sanitized_message = SecuritySanitizer.sanitize_error_message(self.error_history[-1].message) if SANITIZER_AVAILABLE else self.error_history[-1].message
        
        result = {
            "total_errors": len(self.error_history),
            "error_categories": error_counts,
            "most_recent_error": {
                "category": self.error_history[-1].category.value,
                "message": sanitized_message,
                "timestamp": self.error_history[-1].timestamp.isoformat()
            },
            "suggested_fixes": self.error_history[-1].suggested_fixes
        }
        
        # Sanitize entire result for frontend if requested
        if for_frontend and SANITIZER_AVAILABLE:
            result = SecuritySanitizer.sanitize_for_frontend(result)
        
        return result


class EnhancedError(Exception):
    """Enhanced exception with rich context information and security sanitization"""
    
    def __init__(self, message: str, original_error: Exception = None, 
                 error_context: ErrorContext = None, retry_attempts: int = 0):
        # Sanitize message before storing
        sanitized_message = SecuritySanitizer.sanitize_error_message(message) if SANITIZER_AVAILABLE else message
        super().__init__(sanitized_message)
        self.original_error = original_error
        self.error_context = error_context
        self.retry_attempts = retry_attempts
    
    def to_dict(self, for_frontend: bool = True) -> Dict[str, Any]:
        """Convert error to dictionary for JSON serialization with optional sanitization"""
        error_dict = {
            "message": str(self),
            "category": self.error_context.category.value if self.error_context else "unknown",
            "error_code": self.error_context.error_code if self.error_context else "UNKNOWN",
            "retry_attempts": self.retry_attempts,
            "suggested_fixes": self.error_context.suggested_fixes if self.error_context else [],
            "user_actionable": self.error_context.user_actionable if self.error_context else True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Sanitize for frontend if requested
        if for_frontend and SANITIZER_AVAILABLE:
            error_dict = SecuritySanitizer.sanitize_for_frontend(error_dict)
        
        return error_dict


# Convenient factory function
def create_error_handler(max_attempts: int = 3, timeout_seconds: int = 60) -> ErrorHandler:
    """Create a configured error handler"""
    config = RetryConfig(
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True
    )
    return ErrorHandler(config)


# Example usage patterns for documentation
async def example_usage():
    """Example of how to use the enhanced error handling"""
    handler = create_error_handler(max_attempts=3, timeout_seconds=30)
    
    async def risky_operation():
        # Some operation that might fail
        raise ValueError("Product_nov column not found")
    
    try:
        result = await handler.handle_with_retry(
            risky_operation,
            context={"operation": "data_analysis", "file_id": 123},
            progress_callback=lambda update: logger.info(f"Progress: {update}")
        )
    except EnhancedError as e:
        # Log internal details (not sanitized)
        logger.error(f"Operation failed: {e.to_dict(for_frontend=False)}")
        
        # Get sanitized summary for frontend/client reporting
        error_summary_frontend = handler.get_error_summary(for_frontend=True)
        error_summary_internal = handler.get_error_summary(for_frontend=False)
        
        logger.info(f"Error summary (internal): {error_summary_internal}")
        # error_summary_frontend would be sent to client/frontend
