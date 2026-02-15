"""Multi-account management and per-account path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from accounts.account_state_handler import AccountStateHandler
from core.constants import ACCOUNT_METRICS_FILENAME, STRATEGY_MODEL_FILENAME, UPLOAD_QUEUE_FILENAME
from core.config_loader import active_accounts


@dataclass(frozen=True)
class AccountProfile:
    """Immutable account profile from configuration."""

    name: str
    niche: str
    language: str
    timezone: str
    ollama_model: str
    tags_seed: list[str]
    session_state_path: Path
    active: bool = True


class AccountManager:
    """Manage account profiles and isolated account storage."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        storage_root = Path(config["project"]["storage_root"])
        self.storage_root = storage_root if storage_root.is_absolute() else project_root / storage_root
        self.accounts_root = self.storage_root / "accounts"

    def list_accounts(self, only_active: bool = True) -> list[AccountProfile]:
        """Return account profiles."""
        source = active_accounts(self.config) if only_active else self.config.get("accounts", [])
        return [self._to_profile(account) for account in source]

    def get_account(self, account_name: str) -> AccountProfile:
        """Get account profile by name."""
        for account in self.config.get("accounts", []):
            if account.get("name") == account_name:
                return self._to_profile(account)
        raise KeyError(f"Account not found: {account_name}")

    def ensure_account_dirs(self, account: AccountProfile) -> Path:
        """Ensure account-specific storage directories exist."""
        account_dir = self.accounts_root / account.name
        account_dir.mkdir(parents=True, exist_ok=True)
        (account_dir / "videos").mkdir(exist_ok=True)
        (account_dir / "tmp").mkdir(exist_ok=True)
        return account_dir

    def state_handler(self, account: AccountProfile) -> AccountStateHandler:
        """Get account state handler."""
        account_dir = self.ensure_account_dirs(account)
        return AccountStateHandler(account_dir)

    def strategy_model_path(self, account: AccountProfile) -> Path:
        """Path to strategy model JSON."""
        account_dir = self.ensure_account_dirs(account)
        return account_dir / STRATEGY_MODEL_FILENAME

    def metrics_path(self, account: AccountProfile) -> Path:
        """Path to account metrics JSONL."""
        account_dir = self.ensure_account_dirs(account)
        return account_dir / ACCOUNT_METRICS_FILENAME

    def queue_path(self, account: AccountProfile) -> Path:
        """Path to account upload queue."""
        account_dir = self.ensure_account_dirs(account)
        return account_dir / UPLOAD_QUEUE_FILENAME

    def account_video_dir(self, account: AccountProfile) -> Path:
        """Directory for generated account video artifacts."""
        account_dir = self.ensure_account_dirs(account)
        videos_dir = account_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        return videos_dir

    def _to_profile(self, account: dict[str, Any]) -> AccountProfile:
        session_state = Path(account.get("session_state_path", ""))
        if not session_state.is_absolute():
            session_state = self.project_root / session_state

        return AccountProfile(
            name=account["name"],
            niche=account.get("niche", "general"),
            language=account.get("language", "en"),
            timezone=account.get("timezone", "UTC"),
            ollama_model=account.get("ollama_model", "llama3.1:8b"),
            tags_seed=list(account.get("tags_seed", [])),
            session_state_path=session_state,
            active=bool(account.get("active", True)),
        )
