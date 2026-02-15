"""Performance analyzer with configurable engagement scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class EngagementWeights:
    """Configurable weights for engagement score."""

    views: float
    completion_rate: float
    shares: float
    comments: float


class PerformanceAnalyzer:
    """Compute engagement scores and enriched metrics."""

    def __init__(self, weights: EngagementWeights) -> None:
        self.weights = weights

    def engagement_score(self, row: dict[str, Any]) -> float:
        """Calculate score using required formula."""
        return (
            self.weights.views * float(row.get("views", 0))
            + self.weights.completion_rate * float(row.get("completion_rate", 0.0))
            + self.weights.shares * float(row.get("shares", 0))
            + self.weights.comments * float(row.get("comments", 0))
        )

    def analyze(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Compute engagement scores for all rows."""
        if metrics.empty:
            return metrics

        scored = metrics.copy()
        scored["engagement_score"] = scored.apply(
            lambda row: self.engagement_score(row.to_dict()), axis=1
        )
        scored["score_rank_pct"] = scored["engagement_score"].rank(pct=True, method="average")
        return scored.sort_values("engagement_score", ascending=False)
