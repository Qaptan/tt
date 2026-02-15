"""Structured logging utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from core.constants import DEFAULT_LOG_LEVEL, DEFAULT_LOG_NAME


class JsonFormatter(logging.Formatter):
    """JSON line log formatter for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "context"):
            payload["context"] = getattr(record, "context")
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that supports structured context."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        context = kwargs.pop("context", None)
        if context is not None:
            extra["context"] = context
        return msg, kwargs


def setup_logger(log_dir: Path, name: str = DEFAULT_LOG_NAME, level: str = DEFAULT_LOG_LEVEL) -> ContextAdapter:
    """Create console + JSONL file logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    console = RichHandler(rich_tracebacks=True)
    console.setLevel(level.upper())
    console.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_dir / f"{name}.jsonl", encoding="utf-8")
    file_handler.setLevel(level.upper())
    file_handler.setFormatter(JsonFormatter())

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return ContextAdapter(logger, {})
