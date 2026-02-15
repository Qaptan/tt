"""Publish time optimization utilities."""

from __future__ import annotations

import pandas as pd


class PublishTimeOptimizer:
    """Select best publish hour from historical score data."""

    def best_hour(self, scored_df: pd.DataFrame, fallback_hour: int = 12) -> int:
        """Compute best hour based on average engagement score."""
        if scored_df.empty or "publish_hour" not in scored_df or "engagement_score" not in scored_df:
            return fallback_hour

        grouped = scored_df.groupby("publish_hour", as_index=False)["engagement_score"].mean()
        if grouped.empty:
            return fallback_hour
        best = grouped.sort_values("engagement_score", ascending=False).iloc[0]
        return int(best["publish_hour"])
