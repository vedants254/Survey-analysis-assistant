"""
Security Sanitizer
Removes sensitive information from error messages and logs before sending to frontend
"""

import re
import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class SecuritySanitizer:
    """Sanitizes sensitive information from errors, logs, and data before frontend exposure"""
    
    # Common API key patterns
    API_KEY_PATTERNS = [
        # OpenAI API keys
        r'sk-[a-zA-Z0-9]{48,}',
        r'sk-proj-[a-zA-Z0-9_-]{32,}',
        
        # Google/Gemini API keys  
        r'AIza[a-zA-Z0-9_-]{35}',
        
        # Anthropic API keys
        r'sk-ant-api03-[a-zA-Z0-9_-]{95}',
        
        # Generic API key patterns
        r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}["\']?',
        r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}["\']?',
        r'secret["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}["\']?',
        
        # Bearer tokens
        r'token:[\s]*bearer_[a-zA-Z0-9_-]{12,}',
        r'Bearer\s+[a-zA-Z0-9_-]{16,}',
        r'bearer_[a-zA-Z0-9_-]{12,}',
        
        # Environment variable references
        r'OPENAI_API_KEY=.*',
        r'GOOGLE_API_KEY=.*',
        r'ANTHROPIC_API_KEY=.*',
        r'GEMINI_API_KEY=.*',
    ]
    
    # Sensitive environment variable names
    SENSITIVE_ENV_VARS = [
        'OPENAI_API_KEY', 'GOOGLE_API_KEY', 'ANTHROPIC_API_KEY', 
        'GEMINI_API_KEY', 'API_KEY', 'SECRET_KEY', 'ACCESS_TOKEN',
        'REFRESH_TOKEN', 'DATABASE_URL', 'DATABASE_PASSWORD'
    ]
    
    # Other sensitive patterns
    SENSITIVE_PATTERNS = [
        # Database connection strings
        r'postgresql://[^@]+:[^@]+@[^/]+/[^?\s]+',
        r'mysql://[^@]+:[^@]+@[^/]+/[^?\s]+',
        r'mongodb://[^@]+:[^@]+@[^/]+/[^?\s]+',
        
        # JWT tokens
        r'eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+',
        
        # Generic tokens/secrets (20+ chars)
        r'["\']?[a-f0-9]{32,}["\']?',  # Hex tokens
        r'["\']?[A-Za-z0-9+/]{20,}={0,2}["\']?',  # Base64 tokens
    ]
    
    @classmethod
    def sanitize_error_message(cls, error_message: str) -> str:
        """Sanitize error message by removing/masking sensitive information"""
        if not error_message:
            return error_message
        
        sanitized = error_message
        
        # Sanitize API keys with specific patterns
        for pattern in cls.API_KEY_PATTERNS:
            sanitized = re.sub(pattern, cls._mask_api_key, sanitized, flags=re.IGNORECASE)
        
        # Sanitize environment variables
        for env_var in cls.SENSITIVE_ENV_VARS:
            sanitized = re.sub(
                rf'{env_var}["\']?\s*[:=]\s*["\']?[^"\'\s]+["\']?',
                f'{env_var}=***REDACTED***',
                sanitized,
                flags=re.IGNORECASE
            )
        
        # Sanitize other sensitive patterns
        for pattern in cls.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, '***REDACTED***', sanitized)
        
        return sanitized
    
    @classmethod
    def _mask_api_key(cls, match) -> str:
        """Mask API key showing only first 8 and last 4 characters"""
        key = match.group(0)
        if len(key) <= 12:
            return f"{key[:4]}***{key[-2:]}"
        else:
            return f"{key[:8]}***{key[-4:]}"
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
        """Recursively sanitize dictionary data"""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        
        for key, value in data.items():
            # Check if key suggests sensitive data
            key_lower = key.lower()
            is_sensitive_key = any(sensitive in key_lower for sensitive in [
                'key', 'token', 'secret', 'password', 'auth', 'credential'
            ])
            
            if is_sensitive_key and isinstance(value, str):
                # Mask sensitive values
                sanitized[key] = cls._mask_sensitive_value(value)
            elif deep and isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = cls.sanitize_dict(value, deep=True)
            elif deep and isinstance(value, list):
                # Sanitize list items
                sanitized[key] = cls.sanitize_list(value, deep=True)
            elif isinstance(value, str):
                # Sanitize string values for embedded sensitive data
                sanitized[key] = cls.sanitize_error_message(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    @classmethod
    def sanitize_list(cls, data: List[Any], deep: bool = True) -> List[Any]:
        """Recursively sanitize list data"""
        if not isinstance(data, list):
            return data
        
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                sanitized.append(cls.sanitize_error_message(item))
            elif deep and isinstance(item, dict):
                sanitized.append(cls.sanitize_dict(item, deep=True))
            elif deep and isinstance(item, list):
                sanitized.append(cls.sanitize_list(item, deep=True))
            else:
                sanitized.append(item)
        
        return sanitized
    
    @classmethod
    def _mask_sensitive_value(cls, value: str) -> str:
        """Mask a sensitive value appropriately"""
        if not value:
            return value
        
        # For very short values, mask completely
        if len(value) <= 8:
            return "***REDACTED***"
        
        # For API key-like values, show beginning and end
        if len(value) >= 20:
            return f"{value[:6]}***{value[-4:]}"
        
        # For medium values, show less
        return f"{value[:3]}***{value[-2:]}"
    
    @classmethod
    def sanitize_traceback(cls, traceback_str: str) -> str:
        """Sanitize traceback information"""
        if not traceback_str:
            return traceback_str
        
        # Split into lines and sanitize each
        lines = traceback_str.split('\n')
        sanitized_lines = []
        
        for line in lines:
            sanitized_line = cls.sanitize_error_message(line)
            sanitized_lines.append(sanitized_line)
        
        return '\n'.join(sanitized_lines)
    
    @classmethod
    def sanitize_for_frontend(cls, data: Any) -> Any:
        """Main method to sanitize any data before sending to frontend"""
        if isinstance(data, str):
            return cls.sanitize_error_message(data)
        elif isinstance(data, dict):
            return cls.sanitize_dict(data, deep=True)
        elif isinstance(data, list):
            return cls.sanitize_list(data, deep=True)
        else:
            return data
    
    @classmethod
    def sanitize_websocket_message(cls, message: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize WebSocket message before sending to frontend"""
        sanitized = cls.sanitize_dict(message, deep=True)
        
        # Extra check for common WebSocket message fields that might contain sensitive data
        if 'error' in sanitized:
            sanitized['error'] = cls.sanitize_error_message(str(sanitized['error']))
        
        if 'traceback' in sanitized:
            sanitized['traceback'] = cls.sanitize_traceback(str(sanitized['traceback']))
        
        if 'error_details' in sanitized:
            sanitized['error_details'] = cls.sanitize_dict(sanitized['error_details'], deep=True)
        
        return sanitized
    
    @classmethod
    def sanitize_log_message(cls, message: str) -> str:
        """Sanitize log message while preserving debugging utility"""
        # For logs, we can be slightly less aggressive since they're not frontend-facing
        # But still remove full API keys
        sanitized = message
        
        # Replace full API keys with truncated versions
        for pattern in cls.API_KEY_PATTERNS:
            sanitized = re.sub(pattern, cls._mask_api_key, sanitized, flags=re.IGNORECASE)
        
        return sanitized


# Decorator for automatic error sanitization
def sanitize_errors(func):
    """Decorator to automatically sanitize errors from function calls"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Sanitize the error message
            sanitized_message = SecuritySanitizer.sanitize_error_message(str(e))
            
            # Create new exception with sanitized message
            sanitized_exception = type(e)(sanitized_message)
            sanitized_exception.__cause__ = e  # Preserve original for debugging
            
            raise sanitized_exception
    
    return wrapper


# Async version of the decorator
def sanitize_errors_async(func):
    """Async decorator to automatically sanitize errors from async function calls"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Sanitize the error message
            sanitized_message = SecuritySanitizer.sanitize_error_message(str(e))
            
            # Log the original error (sanitized) for debugging
            logger.error(f"Sanitized error in {func.__name__}: {sanitized_message}")
            
            # Create new exception with sanitized message
            sanitized_exception = type(e)(sanitized_message)
            sanitized_exception.__cause__ = e  # Preserve original for debugging
            
            raise sanitized_exception
    
    return wrapper


# Example usage and testing
if __name__ == "__main__":
    # Test API key sanitization
    test_messages = [
        "OpenAI API error with key sk-1234567890abcdef1234567890abcdef1234567890abcdef",
        "Google API key AIzaSyBVWzQHphb5V7QzZNrHcJNgYVQ123456789 is invalid",
        "Error: OPENAI_API_KEY=sk-proj-abcd1234567890 not found",
        "Database connection postgresql://user:pass@localhost:5432/db failed",
        "JWT token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c in headers"
    ]
    
    print("🔒 Testing Security Sanitization:")
    for msg in test_messages:
        sanitized = SecuritySanitizer.sanitize_error_message(msg)
        print(f"Original: {msg[:60]}...")
        print(f"Sanitized: {sanitized}")
        print()