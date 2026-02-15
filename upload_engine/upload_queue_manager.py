"""Persistent upload queue manager."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upload_engine.draft_or_publish_mode import DraftOrPublishMode


@dataclass
class UploadTask:
    """Pending upload task."""

    id: str
    account_name: str
    video_path: str
    caption: str
    hashtags: list[str]
    mode: str
    thumbnail_path: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str = ""
    retries: int = 0
    status: str = "pending"


class UploadQueueManager:
    """Manage account-specific upload queue in JSON format."""

    def __init__(self, queue_path: Path) -> None:
        self.queue_path = queue_path
        self.lock = threading.Lock()
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.queue_path.exists():
            self._write({"tasks": []})

    def enqueue(
        self,
        account_name: str,
        video_path: Path,
        caption: str,
        hashtags: list[str],
        mode: DraftOrPublishMode,
        thumbnail_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UploadTask:
        """Add a task to the queue."""
        task = UploadTask(
            id=str(uuid.uuid4()),
            account_name=account_name,
            video_path=str(video_path),
            caption=caption,
            hashtags=hashtags,
            mode=mode.value,
            thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            retries=0,
            status="pending",
        )
        with self.lock:
            payload = self._read()
            payload["tasks"].append(asdict(task))
            self._write(payload)
        return task

    def next_pending(self) -> UploadTask | None:
        """Return and reserve next pending task."""
        with self.lock:
            payload = self._read()
            for idx, raw in enumerate(payload.get("tasks", [])):
                if raw.get("status") != "pending":
                    continue
                raw["status"] = "in_progress"
                payload["tasks"][idx] = raw
                self._write(payload)
                return UploadTask(**raw)
        return None

    def mark_completed(self, task_id: str, post_id: str | None = None) -> None:
        """Mark task as completed."""
        with self.lock:
            payload = self._read()
            for task in payload.get("tasks", []):
                if task.get("id") == task_id:
                    task["status"] = "completed"
                    if post_id:
                        task["post_id"] = post_id
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self._write(payload)

    def mark_failed(self, task_id: str, reason: str) -> None:
        """Mark task as failed and keep reason."""
        with self.lock:
            payload = self._read()
            for task in payload.get("tasks", []):
                if task.get("id") == task_id:
                    task["status"] = "failed"
                    task["error"] = reason
                    task["failed_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self._write(payload)

    def increment_retry(self, task_id: str) -> None:
        """Increment retry counter and return task to pending state."""
        with self.lock:
            payload = self._read()
            for task in payload.get("tasks", []):
                if task.get("id") == task_id:
                    task["retries"] = int(task.get("retries", 0)) + 1
                    task["status"] = "pending"
                    break
            self._write(payload)

    def pending_count(self) -> int:
        """Number of pending tasks."""
        payload = self._read()
        return sum(1 for task in payload.get("tasks", []) if task.get("status") == "pending")

    def all_tasks(self) -> list[dict[str, Any]]:
        """Return all queue tasks."""
        return self._read().get("tasks", [])

    def _read(self) -> dict[str, Any]:
        return json.loads(self.queue_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.queue_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
