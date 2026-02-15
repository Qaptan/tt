"""Trend scraping module.

Uses a local mock strategy by default so the system can run offline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class TrendSignal:
    """Raw trend signal for downstream scoring."""

    phrase: str
    niche: str
    source: str
    mentions_growth: float
    competition: float
    freshness_hours: int


class TrendScraper:
    """Scrape trends using local heuristics and curated pools."""

    BASE_TRENDS: dict[str, list[str]] = {
        "fitness": [
            "7 minute core blast",
            "high protein breakfast hacks",
            "mobility before bed",
            "fat loss myths",
            "dumbbell only routine",
        ],
        "personal finance": [
            "50/30/20 budget breakdown",
            "stop impulse spending",
            "high yield savings explained",
            "credit utilization mistakes",
            "paycheck automation plan",
        ],
        "general": [
            "3 mistakes everyone makes",
            "daily habit that compounds",
            "before you start this",
            "one minute checklist",
            "hidden trick nobody uses",
        ],
    }

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def scrape(self, niche: str, limit: int = 10) -> list[TrendSignal]:
        """Return mock trend signals for a niche."""
        key = niche.lower().strip()
        candidates = self.BASE_TRENDS.get(key, self.BASE_TRENDS["general"])
        selected = [self._random.choice(candidates) for _ in range(max(limit, 1))]

        now = datetime.now(timezone.utc)
        return [
            TrendSignal(
                phrase=phrase,
                niche=niche,
                source="local_mock",
                mentions_growth=round(self._random.uniform(0.8, 2.4), 3),
                competition=round(self._random.uniform(0.2, 0.95), 3),
                freshness_hours=int((now.minute + i * 7) % 72),
            )
            for i, phrase in enumerate(selected)
        ]
