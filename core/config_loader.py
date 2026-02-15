"""Configuration loading and validation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from core.constants import DEFAULT_CONFIG_PATH, DEFAULT_ENV_PATH, DEFAULT_STORAGE_ROOT


class ConfigError(RuntimeError):
    """Raised when configuration is invalid."""


class ConfigLoader:
    """Load YAML config with optional .env interpolation and defaults."""

    def __init__(self, config_path: Path | None = None, env_path: Path | None = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.env_path = env_path or DEFAULT_ENV_PATH

    def load(self) -> dict[str, Any]:
        """Load and normalize configuration."""
        self._load_env_file()
        if not self.config_path.exists():
            raise ConfigError(f"Config not found: {self.config_path}")

        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        config = self._interpolate_env(raw)
        self._validate(config)
        self._apply_defaults(config)
        self._ensure_storage_dirs(config)
        return config

    def _load_env_file(self) -> None:
        """Load key/value pairs from .env if present."""
        if not self.env_path.exists():
            return
        for line in self.env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    def _interpolate_env(self, data: Any) -> Any:
        """Recursively interpolate ${VAR} values from environment."""
        if isinstance(data, dict):
            return {k: self._interpolate_env(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._interpolate_env(v) for v in data]
        if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_key = data[2:-1]
            return os.getenv(env_key, "")
        return data

    def _validate(self, config: dict[str, Any]) -> None:
        """Validate minimum required sections."""
        required = [
            "project",
            "execution",
            "generation",
            "analytics",
            "strategy",
            "upload",
            "scheduler",
            "accounts",
        ]
        missing = [key for key in required if key not in config]
        if missing:
            raise ConfigError(f"Missing required config sections: {', '.join(missing)}")
        if not isinstance(config.get("accounts"), list) or not config["accounts"]:
            raise ConfigError("Config 'accounts' must be a non-empty list")

    def _apply_defaults(self, config: dict[str, Any]) -> None:
        """Apply defaults for optional values."""
        project = config.setdefault("project", {})
        project.setdefault("storage_root", str(DEFAULT_STORAGE_ROOT))

        execution = config.setdefault("execution", {})
        execution.setdefault("account_mode", "sequential")
        execution.setdefault("max_workers", 1)

        upload = config.setdefault("upload", {})
        upload.setdefault("retry_attempts", 3)
        upload.setdefault("retry_backoff_seconds", 5)
        upload.setdefault("mock_mode", True)

        scheduler = config.setdefault("scheduler", {})
        scheduler.setdefault("interval_minutes", 60)

    def _ensure_storage_dirs(self, config: dict[str, Any]) -> None:
        """Ensure root storage directories exist."""
        storage_root = Path(config["project"]["storage_root"])
        if not storage_root.is_absolute():
            storage_root = self.config_path.parent / storage_root
        for relative in ["videos", "logs", "metrics", "models", "accounts", "queue"]:
            (storage_root / relative).mkdir(parents=True, exist_ok=True)


def get_account_config(config: dict[str, Any], account_name: str) -> dict[str, Any]:
    """Return account config by name."""
    for account in config.get("accounts", []):
        if account.get("name") == account_name:
            return account
    raise ConfigError(f"Account not found in config: {account_name}")


def active_accounts(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return active accounts from config."""
    return [a for a in config.get("accounts", []) if a.get("active", True)]
