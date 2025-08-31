import re

def sanitize_error(msg: str) -> str:
    """Redact sensitive information from error messages"""
    # Redact API keys (common patterns)
    msg = re.sub(r"(?i)(api[_-]?key|token|secret)[=:][^\s]+", r"\1=***", msg)
    # Redact JWT tokens
    msg = re.sub(r"\beyJ[\w-]+\.eyJ[\w-]+\.\w+\b", "***", msg)
    # Redact long hex strings
    msg = re.sub(r"\b[a-f0-9]{24,}\b", "***", msg)
    return msg

