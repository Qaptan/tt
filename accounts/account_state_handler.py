"""Persistent per-account state handling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.constants import ACCOUNT_STATE_FILENAME


@dataclass
class AccountState:
    """Mutable account-level runtime and learning state."""

    account_name: str
    generated_count: int = 0
    uploaded_count: int = 0
    failed_uploads: int = 0
    pending_metrics_post_ids: list[str] = field(default_factory=list)
    last_run_at: str | None = None
    last_success_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert state to serializable dict."""
        return {
            "account_name": self.account_name,
            "generated_count": self.generated_count,
            "uploaded_count": self.uploaded_count,
            "failed_uploads": self.failed_uploads,
            "pending_metrics_post_ids": self.pending_metrics_post_ids,
            "last_run_at": self.last_run_at,
            "last_success_at": self.last_success_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AccountState":
        """Create state from stored dictionary."""
        return cls(
            account_name=payload.get("account_name", "unknown"),
            generated_count=int(payload.get("generated_count", 0)),
            uploaded_count=int(payload.get("uploaded_count", 0)),
            failed_uploads=int(payload.get("failed_uploads", 0)),
            pending_metrics_post_ids=list(payload.get("pending_metrics_post_ids", [])),
            last_run_at=payload.get("last_run_at"),
            last_success_at=payload.get("last_success_at"),
        )


class AccountStateHandler:
    """Read and write account state from account_state.json."""

    def __init__(self, account_dir: Path) -> None:
        self.account_dir = account_dir
        self.path = account_dir / ACCOUNT_STATE_FILENAME

    def load(self, account_name: str) -> AccountState:
        """Load account state or initialize defaults."""
        if not self.path.exists():
            state = AccountState(account_name=account_name)
            self.save(state)
            return state

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        state = AccountState.from_dict(payload)
        if state.account_name == "unknown":
            state.account_name = account_name
        return state

    def save(self, state: AccountState) -> None:
        """Persist account state as JSON."""
        self.account_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

    def mark_run(self, state: AccountState, success: bool) -> AccountState:
        """Update run metadata after a processing cycle."""
        now = datetime.now(timezone.utc).isoformat()
        state.last_run_at = now
        if success:
            state.last_success_at = now
        self.save(state)
        return state
