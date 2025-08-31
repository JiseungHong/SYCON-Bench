
import re

def sanitize_error_message(msg: str) -> str:
    """Redact sensitive information from error messages"""
    # Redact API keys (common formats)
    msg = re.sub(r'(?i)(api[_-]?key|token|secret)[=:]\s*[\'"]?[a-zA-Z0-9-]{24,}[\'"]?', r'\1=***', msg)
    # Redact JWT tokens
    msg = re.sub(r'\beyJ[\w-]+\.eyJ[\w-]+\.\w+\b', '***', msg)
    # Redact long hexadecimal strings
    msg = re.sub(r'\b[a-f0-9]{16,}\b', '***', msg)
    return msg
