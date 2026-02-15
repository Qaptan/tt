"""Winner and loser pattern detection."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


class WinnerPatternDetector:
    """Detect patterns from top and bottom scoring content."""

    def __init__(self, winner_quantile: float = 0.75, loser_quantile: float = 0.25) -> None:
        self.winner_quantile = winner_quantile
        self.loser_quantile = loser_quantile

    def detect(self, scored_df: pd.DataFrame) -> dict[str, Any]:
        """Return winner and loser patterns."""
        if scored_df.empty or "engagement_score" not in scored_df:
            return {
                "winners": {"hooks": [], "durations": [], "caption_styles": [], "formats": [], "publish_hours": []},
                "losers": {"hooks": [], "durations": [], "caption_styles": [], "formats": [], "publish_hours": []},
            }

        high_cut = scored_df["engagement_score"].quantile(self.winner_quantile)
        low_cut = scored_df["engagement_score"].quantile(self.loser_quantile)

        winners = scored_df[scored_df["engagement_score"] >= high_cut]
        losers = scored_df[scored_df["engagement_score"] <= low_cut]

        return {
            "winners": {
                "hooks": self._top_values(winners, "hook_style"),
                "durations": self._top_values(winners, "duration_seconds"),
                "caption_styles": self._top_values(winners, "caption_style"),
                "formats": self._top_values(winners, "format_name"),
                "publish_hours": self._top_values(winners, "publish_hour"),
            },
            "losers": {
                "hooks": self._top_values(losers, "hook_style"),
                "durations": self._top_values(losers, "duration_seconds"),
                "caption_styles": self._top_values(losers, "caption_style"),
                "formats": self._top_values(losers, "format_name"),
                "publish_hours": self._top_values(losers, "publish_hour"),
            },
        }

    def _top_values(self, frame: pd.DataFrame, column: str, top_n: int = 3) -> list[Any]:
        if frame.empty or column not in frame:
            return []
        counts = Counter(frame[column].tolist())
        return [key for key, _ in counts.most_common(top_n)]
