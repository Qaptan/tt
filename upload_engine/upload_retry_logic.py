"""Retry logic utilities for upload operations."""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_operation(
    operation: Callable[[], T],
    attempts: int,
    backoff_seconds: int,
    logger: Any,
    context: dict[str, Any] | None = None,
) -> T:
    """Execute operation with fixed-backoff retry behavior."""
    if attempts < 1:
        attempts = 1

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Upload attempt failed",
                context={
                    **(context or {}),
                    "attempt": attempt,
                    "attempts": attempts,
                    "error": str(exc),
                },
            )
            if attempt < attempts:
                time.sleep(max(0, backoff_seconds))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry operation failed without exception")
