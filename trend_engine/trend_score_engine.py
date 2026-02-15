"""Trend scoring engine."""

from __future__ import annotations

from dataclasses import dataclass

from trend_engine.trend_classifier import ClassifiedTrend


@dataclass
class ScoredTrend:
    """Trend with normalized score for ranking."""

    phrase: str
    category: str
    intent: str
    score: float


class TrendScoreEngine:
    """Compute weighted trend scores."""

    def __init__(self, growth_weight: float = 0.5, freshness_weight: float = 0.3, competition_weight: float = 0.2) -> None:
        self.growth_weight = growth_weight
        self.freshness_weight = freshness_weight
        self.competition_weight = competition_weight

    def score(self, trends: list[ClassifiedTrend]) -> list[ScoredTrend]:
        """Rank trends by opportunity score."""
        scored: list[ScoredTrend] = []
        for trend in trends:
            freshness_score = max(0.0, 1.0 - (trend.freshness_hours / 72.0))
            competition_score = max(0.0, 1.0 - trend.competition)
            normalized_growth = min(1.0, trend.mentions_growth / 2.5)
            total = (
                self.growth_weight * normalized_growth
                + self.freshness_weight * freshness_score
                + self.competition_weight * competition_score
            )
            scored.append(
                ScoredTrend(
                    phrase=trend.phrase,
                    category=trend.category,
                    intent=trend.intent,
                    score=round(total, 4),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)
