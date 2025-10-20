"""
Security utilities for SYCON-Bench.

This module provides utilities to prevent API keys and other sensitive data
from being exposed in logs, error messages, and debug output.
"""
import re
import logging
from typing import Any, Dict, Union


class SecureLogger:
    """A secure logger that sanitizes sensitive data before logging."""

    # Patterns that might indicate API keys or sensitive data
    SENSITIVE_PATTERNS = [
        # OpenAI API keys
        r'sk-[a-zA-Z0-9]{48}',
        r'sk-proj-[a-zA-Z0-9]{48}',
        # Anthropic API keys
        r'sk-ant-[a-zA-Z0-9\-]{95}',
        # Generic API key patterns
        r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]{15,}["\']?',
        r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]{15,}["\']?',
        # Authorization headers
        r'authorization["\']?\s*[:=]\s*["\']?bearer\s+[a-zA-Z0-9\-_\.]{15,}',
        # Generic secrets (32+ chars of alphanumeric/special chars, but not common words)
        r'(?<![\w\-])[a-zA-Z0-9\-_\.]{32,}(?![\w\-])',
    ]

    @staticmethod
    def sanitize_text(text: str) -> str:
        """
        Sanitize text by replacing potential API keys and sensitive data with placeholders.

        Args:
            text: The text to sanitize

        Returns:
            Sanitized text with sensitive data replaced
        """
        if not isinstance(text, str):
            text = str(text)

        sanitized = text

        # Replace potential API keys with placeholders
        for pattern in SecureLogger.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)

        return sanitized

    @staticmethod
    def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a dictionary by redacting sensitive keys and values.

        Args:
            data: Dictionary to sanitize

        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            return data

        sanitized = {}
        sensitive_keys = {'api_key', 'token', 'authorization', 'secret', 'password', 'key'}

        for key, value in data.items():
            key_lower = key.lower()

            # Check if key name suggests sensitive data
            if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                sanitized[key] = '[REDACTED]' if value else None
            elif isinstance(value, str):
                sanitized[key] = SecureLogger.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[key] = SecureLogger.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [SecureLogger.sanitize_text(str(item)) if isinstance(item, str) else item for item in value]
            else:
                sanitized[key] = value

        return sanitized

    @staticmethod
    def sanitize_exception(exception: Exception) -> str:
        """
        Sanitize exception message to remove potential sensitive data.

        Args:
            exception: The exception to sanitize

        Returns:
            Sanitized exception message
        """
        exc_str = str(exception)
        return SecureLogger.sanitize_text(exc_str)

    @staticmethod
    def secure_log(level: int, message: str, *args, **kwargs):
        """
        Log a message after sanitizing it for sensitive data.

        Args:
            level: Logging level (e.g., logging.INFO)
            message: Message to log
            *args: Additional arguments for logging
            **kwargs: Additional keyword arguments for logging
        """
        sanitized_message = SecureLogger.sanitize_text(message)
        # Get the logger and log with proper formatting
        logger = logging.getLogger()
        logger.log(level, sanitized_message, *args, **kwargs)

    @staticmethod
    def secure_debug(message: str, *args, **kwargs):
        """Log a debug message securely."""
        SecureLogger.secure_log(logging.DEBUG, message, *args, **kwargs)

    @staticmethod
    def secure_info(message: str, *args, **kwargs):
        """Log an info message securely."""
        SecureLogger.secure_log(logging.INFO, message, *args, **kwargs)

    @staticmethod
    def secure_warning(message: str, *args, **kwargs):
        """Log a warning message securely."""
        SecureLogger.secure_log(logging.WARNING, message, *args, **kwargs)

    @staticmethod
    def secure_error(message: str, *args, **kwargs):
        """Log an error message securely."""
        SecureLogger.secure_log(logging.ERROR, message, *args, **kwargs)


def sanitize_prompt_for_logging(prompt: str, max_length: int = 100) -> str:
    """
    Sanitize a prompt for safe logging by truncating and removing sensitive data.

    Args:
        prompt: The prompt to sanitize
        max_length: Maximum length of the logged prompt

    Returns:
        Sanitized and truncated prompt safe for logging
    """
    if not prompt:
        return ""

    # First sanitize sensitive data
    sanitized = SecureLogger.sanitize_text(prompt)

    # Then truncate if needed
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized


def secure_exception_handler(func):
    """
    Decorator to handle exceptions securely by sanitizing error messages.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with secure exception handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the sanitized exception
            sanitized_error = SecureLogger.sanitize_exception(e)
            # Log directly to avoid double sanitization of function name
            logging.error(f"Error in {func.__name__}: {sanitized_error}")
            raise

    return wrapper
