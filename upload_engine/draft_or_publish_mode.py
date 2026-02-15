"""Upload mode definitions."""

from __future__ import annotations

from enum import Enum


class DraftOrPublishMode(str, Enum):
    """Upload destination mode."""

    DRAFT = "draft"
    PUBLISH = "publish"

    @classmethod
    def from_value(cls, value: str) -> "DraftOrPublishMode":
        normalized = (value or "draft").strip().lower()
        if normalized == cls.PUBLISH.value:
            return cls.PUBLISH
        return cls.DRAFT
