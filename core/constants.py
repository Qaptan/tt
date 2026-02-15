"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_STORAGE_ROOT = PROJECT_ROOT / "storage"

ACCOUNT_STATE_FILENAME = "account_state.json"
STRATEGY_MODEL_FILENAME = "strategy_model.json"
ACCOUNT_METRICS_FILENAME = "metrics.jsonl"
UPLOAD_QUEUE_FILENAME = "upload_queue.json"

VIDEO_EXTENSION = ".mp4"
AUDIO_EXTENSION = ".wav"
SUBTITLE_EXTENSION = ".srt"
THUMBNAIL_EXTENSION = ".png"

DRAFT_MODE = "draft"
PUBLISH_MODE = "publish"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_NAME = "autonomous_engine"

FFMPEG_BINARY = "ffmpeg"
WHISPER_BINARY = "whisper"
PIPER_BINARY = "piper"
OLLAMA_BINARY = "ollama"

SECONDS_IN_MINUTE = 60
