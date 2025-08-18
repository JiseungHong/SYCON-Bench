"""
Secure logging utilities for SYCON-Bench.

This module provides utilities to prevent API keys and other sensitive data
from being exposed in logs and error messages.
"""
import re
import logging
from typing import Any, Dict, List, Optional


class SecureLogger:
    """Secure logging utilities to prevent sensitive data exposure."""

    # Patterns that might indicate sensitive data
    SENSITIVE_PATTERNS = [
        r'sk-[a-zA-Z0-9]{48}',  # OpenAI API keys
        r'sk-[a-zA-Z0-9-]{20,}',  # Generic API keys starting with sk-
        r'Bearer [a-zA-Z0-9-_\.]{20,}',  # Bearer tokens
        r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9-_]{20,}',  # API key patterns
        r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9-_\.]{20,}',  # Token patterns
        r'password["\']?\s*[:=]\s*["\']?[^\s"\']{8,}',  # Password patterns
        r'secret["\']?\s*[:=]\s*["\']?[a-zA-Z0-9-_]{20,}',  # Secret patterns
        r'super_secret_password123',  # Test password pattern
    ]

    @classmethod
    def sanitize_string(cls, text: str, replacement: str = "***") -> str:
        """
        Sanitize a string by replacing sensitive patterns with a replacement string.

        Args:
            text: The text to sanitize
            replacement: The replacement string for sensitive data

        Returns:
            Sanitized text with sensitive patterns replaced
        """
        if not isinstance(text, str):
            text = str(text)

        sanitized = text
        for pattern in cls.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized

    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], replacement: str = "***") -> Dict[str, Any]:
        """
        Sanitize a dictionary by replacing sensitive values.

        Args:
            data: Dictionary to sanitize
            replacement: The replacement string for sensitive data

        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            return data

        sanitized = {}
        sensitive_keys = {'api_key', 'token', 'password', 'secret', 'key', 'auth'}

        for key, value in data.items():
            key_lower = key.lower()

            # Check if key name suggests sensitive data
            if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                sanitized[key] = replacement if value else None
            elif isinstance(value, str):
                sanitized[key] = cls.sanitize_string(value, replacement)
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value, replacement)
            elif isinstance(value, list):
                sanitized[key] = [cls.sanitize_string(str(item), replacement) if isinstance(item, str) else item for item in value]
            else:
                sanitized[key] = value

        return sanitized

    @classmethod
    def sanitize_exception(cls, exception: Exception, replacement: str = "***") -> str:
        """
        Sanitize exception message to remove sensitive data.

        Args:
            exception: The exception to sanitize
            replacement: The replacement string for sensitive data

        Returns:
            Sanitized exception message
        """
        error_msg = str(exception)
        return cls.sanitize_string(error_msg, replacement)

    @classmethod
    def safe_log_args(cls, args: Any, logger: Optional[logging.Logger] = None) -> None:
        """
        Safely log command line arguments or configuration.

        Args:
            args: Arguments object (typically from argparse)
            logger: Logger instance to use (defaults to root logger)
        """
        if logger is None:
            logger = logging.getLogger()

        if hasattr(args, '__dict__'):
            log_args = cls.sanitize_dict(vars(args).copy())
        else:
            log_args = cls.sanitize_dict(args if isinstance(args, dict) else {})

        logger.info(f"Arguments: {log_args}")

    @classmethod
    def safe_log_error(cls, message: str, exception: Optional[Exception] = None,
                      logger: Optional[logging.Logger] = None) -> None:
        """
        Safely log an error message with optional exception details.

        Args:
            message: Base error message
            exception: Optional exception to include (will be sanitized)
            logger: Logger instance to use (defaults to root logger)
        """
        if logger is None:
            logger = logging.getLogger()

        if exception:
            sanitized_error = cls.sanitize_exception(exception)
            logger.error(f"{message}: {sanitized_error}")
        else:
            logger.error(message)

    @classmethod
    def safe_print_prompt(cls, prompt: str, max_length: int = 100,
                         prefix: str = "Processing prompt") -> None:
        """
        Safely print a prompt preview without exposing sensitive data.

        Args:
            prompt: The prompt to print
            max_length: Maximum length to show
            prefix: Prefix for the print statement
        """
        if not prompt:
            print(f"{prefix}: [empty]")
            return

        # Sanitize the prompt first
        sanitized_prompt = cls.sanitize_string(prompt)

        # Truncate to max length
        if len(sanitized_prompt) > max_length:
            preview = sanitized_prompt[:max_length] + "..."
        else:
            preview = sanitized_prompt

        print(f"{prefix}: {preview}")

    @classmethod
    def create_safe_error_response(cls, base_message: str, exception: Exception) -> str:
        """
        Create a safe error response that doesn't expose sensitive data.

        Args:
            base_message: Base error message
            exception: The exception that occurred

        Returns:
            Safe error response string
        """
        sanitized_error = cls.sanitize_exception(exception)
        return f"{base_message}: {sanitized_error}"


# Convenience functions for backward compatibility
def sanitize_args_for_logging(args: Any) -> Dict[str, Any]:
    """Sanitize arguments for safe logging."""
    if hasattr(args, '__dict__'):
        return SecureLogger.sanitize_dict(vars(args).copy())
    return SecureLogger.sanitize_dict(args if isinstance(args, dict) else {})


def safe_error_message(exception: Exception) -> str:
    """Create a safe error message from an exception."""
    return SecureLogger.sanitize_exception(exception)
