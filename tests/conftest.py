"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def sample_config_dict(tmp_path: Path) -> dict:
    """Return a reusable config dictionary."""
    return {
        "project": {"name": "test", "storage_root": str(tmp_path / "storage")},
        "execution": {"account_mode": "sequential", "max_workers": 2},
        "generation": {
            "allowed_durations": [9, 12, 15],
            "default_duration_seconds": 12,
            "video": {"resolution": "1080x1920", "fps": 30},
        },
        "analytics": {
            "mock_mode": True,
            "min_age_minutes_for_collection": 0,
            "engagement_weights": {
                "views": 0.001,
                "completion_rate": 100.0,
                "shares": 6.0,
                "comments": 4.0,
            },
        },
        "strategy": {
            "learning_rate": 0.15,
            "underperforming_threshold": 0.45,
            "rotation_floor": 0.05,
            "initial_format_weights": {"story": 1.0, "listicle": 1.0},
            "initial_hook_weights": {"question": 1.0, "shock": 1.0},
            "initial_caption_weights": {"direct": 1.0, "curiosity": 1.0},
            "initial_duration_weights": {"9": 1.0, "12": 1.0},
        },
        "upload": {
            "mode": "draft",
            "retry_attempts": 3,
            "retry_backoff_seconds": 0,
            "headless": True,
            "mock_mode": True,
            "tiktok_upload_url": "https://www.tiktok.com/upload",
            "post_wait_seconds": 0,
        },
        "scheduler": {"interval_minutes": 1, "cron_enabled": False},
        "accounts": [
            {
                "name": "acct_one",
                "active": True,
                "niche": "fitness",
                "language": "en",
                "timezone": "UTC",
                "ollama_model": "llama3.1:8b",
                "tags_seed": ["fitness"],
                "session_state_path": str(tmp_path / "storage" / "accounts" / "acct_one" / "session_state.json"),
            }
        ],
    }


@pytest.fixture
def sample_config_file(tmp_path: Path, sample_config_dict: dict) -> Path:
    """Write sample config to temporary file."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(sample_config_dict), encoding="utf-8")
    return path
