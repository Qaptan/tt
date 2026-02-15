"""Underperforming pattern suppression logic."""

from __future__ import annotations

from typing import Any

import pandas as pd


class KillUnderperformingLogic:
    """Identify and reduce weak-performing patterns."""

    def detect_underperformers(self, scored_df: pd.DataFrame, threshold: float) -> dict[str, list[Any]]:
        """Return patterns whose average rank percentile is below threshold."""
        if scored_df.empty or "score_rank_pct" not in scored_df:
            return {"formats": [], "hooks": [], "caption_styles": [], "durations": []}

        under = scored_df[scored_df["score_rank_pct"] < threshold]
        if under.empty:
            return {"formats": [], "hooks": [], "caption_styles": [], "durations": []}

        return {
            "formats": under["format_name"].dropna().astype(str).unique().tolist() if "format_name" in under else [],
            "hooks": under["hook_style"].dropna().astype(str).unique().tolist() if "hook_style" in under else [],
            "caption_styles": under["caption_style"].dropna().astype(str).unique().tolist() if "caption_style" in under else [],
            "durations": under["duration_seconds"].dropna().astype(int).unique().tolist() if "duration_seconds" in under else [],
        }
