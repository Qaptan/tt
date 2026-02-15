"""Hashtag optimization module."""

from __future__ import annotations

import re


class HashtagOptimizer:
    """Generate and rank hashtags from trends and account niche."""

    def generate(
        self,
        niche: str,
        trend_phrases: list[str],
        seed_tags: list[str],
        max_tags: int = 8,
    ) -> list[str]:
        """Return cleaned hashtag list."""
        raw_tags = [niche] + trend_phrases + seed_tags + ["fyp", "tiktoktips"]
        cleaned: list[str] = []
        seen: set[str] = set()

        for tag in raw_tags:
            normalized = re.sub(r"[^a-zA-Z0-9]", "", tag.replace(" ", "")).lower()
            if not normalized:
                continue
            hash_tag = f"#{normalized}"
            if hash_tag in seen:
                continue
            seen.add(hash_tag)
            cleaned.append(hash_tag)
            if len(cleaned) >= max_tags:
                break

        return cleaned
