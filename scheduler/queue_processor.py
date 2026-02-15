"""Upload queue processing logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from upload_engine.upload_queue_manager import UploadQueueManager, UploadTask
from upload_engine.upload_retry_logic import retry_operation


@dataclass
class QueueProcessResult:
    """Queue processing outcome."""

    processed: int = 0
    succeeded: int = 0
    failed: int = 0


class QueueProcessor:
    """Process pending upload tasks with retry handling."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def process(
        self,
        queue_manager: UploadQueueManager,
        upload_fn: Callable[[UploadTask], tuple[bool, str | None, str | None]],
        attempts: int,
        backoff_seconds: int,
        on_success: Callable[[UploadTask, str | None], None] | None = None,
    ) -> QueueProcessResult:
        """Process all pending queue tasks."""
        outcome = QueueProcessResult()

        while True:
            task = queue_manager.next_pending()
            if task is None:
                break

            outcome.processed += 1

            def operation() -> tuple[bool, str | None, str | None]:
                success, post_id, error = upload_fn(task)
                if not success:
                    raise RuntimeError(error or "unknown upload error")
                return success, post_id, error

            try:
                _, post_id, _ = retry_operation(
                    operation=operation,
                    attempts=attempts,
                    backoff_seconds=backoff_seconds,
                    logger=self.logger,
                    context={"task_id": task.id, "account": task.account_name},
                )
                queue_manager.mark_completed(task.id, post_id=post_id)
                if on_success is not None:
                    on_success(task, post_id)
                outcome.succeeded += 1
            except Exception as exc:  # noqa: BLE001
                queue_manager.mark_failed(task.id, reason=str(exc))
                outcome.failed += 1
                self.logger.error(
                    "Queue task permanently failed",
                    context={"task_id": task.id, "account": task.account_name, "error": str(exc)},
                )

        return outcome
