"""Metrics collection and persistence."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class VideoMetric:
    """Metrics for one uploaded video."""

    post_id: str
    account_name: str
    format_name: str
    hook_style: str
    duration_seconds: int
    caption_style: str
    publish_hour: int
    views: int
    completion_rate: float
    shares: int
    comments: int
    status: str
    created_at: str
    collected_at: str | None = None


class MetricsCollector:
    """Track pending and collected metrics in account-local JSONL file."""

    def __init__(self, metrics_path: Path, logger: Any, seed: int | None = None) -> None:
        self.metrics_path = metrics_path
        self.logger = logger
        self._random = random.Random(seed)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.touch(exist_ok=True)

    def record_pending(self, post_id: str, account_name: str, metadata: dict[str, Any]) -> None:
        """Record a newly uploaded post as pending metrics."""
        payload = {
            "post_id": post_id,
            "account_name": account_name,
            "format_name": metadata.get("format_name", "unknown"),
            "hook_style": metadata.get("hook_style", "unknown"),
            "duration_seconds": int(metadata.get("duration_seconds", 12)),
            "caption_style": metadata.get("caption_style", "unknown"),
            "publish_hour": int(metadata.get("publish_hour", 12)),
            "views": 0,
            "completion_rate": 0.0,
            "shares": 0,
            "comments": 0,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collected_at": None,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def collect(self, min_age_minutes: int = 5, mock_mode: bool = True) -> list[VideoMetric]:
        """Collect metrics for eligible pending records."""
        records = self._read_all()
        now = datetime.now(timezone.utc)
        changed = False
        collected: list[VideoMetric] = []

        for record in records:
            if record.get("status") != "pending":
                continue

            created_at = datetime.fromisoformat(record["created_at"])
            if now - created_at < timedelta(minutes=min_age_minutes):
                continue

            if not mock_mode:
                self.logger.info("Metrics unavailable, kept as pending", context={"post_id": record.get("post_id")})
                continue

            record["views"] = int(self._random.randint(300, 12000))
            record["completion_rate"] = round(self._random.uniform(0.25, 0.95), 4)
            record["shares"] = int(self._random.randint(1, 250))
            record["comments"] = int(self._random.randint(0, 180))
            record["status"] = "collected"
            record["collected_at"] = now.isoformat()
            changed = True

            collected.append(VideoMetric(**record))

        if changed:
            self._write_all(records)

        return collected

    def dataframe(self, status_filter: str | None = None) -> pd.DataFrame:
        """Return all metrics as DataFrame."""
        records = self._read_all()
        if status_filter:
            records = [r for r in records if r.get("status") == status_filter]
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def _read_all(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    def _write_all(self, records: list[dict[str, Any]]) -> None:
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
