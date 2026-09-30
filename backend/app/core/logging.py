"""
FaceVital AI — Structured Logging
==================================
JSON-structured logging for observability.
"""

import logging
import json
import sys
import time
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Outputs log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Merge any extra structured fields
        if hasattr(record, "structured_data"):
            log_entry.update(record.structured_data)
        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO", structured: bool = True) -> None:
    """Configure application-wide logging."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    message: str,
    level: str = "info",
    **kwargs: Any,
) -> None:
    """Log a structured event with additional key-value pairs."""
    extra = {"structured_data": kwargs} if kwargs else {}
    getattr(logger, level)(message, extra=extra)
