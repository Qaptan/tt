"""Trend classification module."""

from __future__ import annotations

from dataclasses import dataclass

from trend_engine.trend_scraper import TrendSignal


@dataclass
class ClassifiedTrend:
    """Trend enriched with category and intent."""

    phrase: str
    niche: str
    source: str
    mentions_growth: float
    competition: float
    freshness_hours: int
    category: str
    intent: str


class TrendClassifier:
    """Classifies trends by lexical heuristics."""

    def classify(self, trends: list[TrendSignal]) -> list[ClassifiedTrend]:
        """Classify trend type and user intent."""
        classified: list[ClassifiedTrend] = []
        for trend in trends:
            text = trend.phrase.lower()
            category = self._infer_category(text)
            intent = self._infer_intent(text)
            classified.append(
                ClassifiedTrend(
                    phrase=trend.phrase,
                    niche=trend.niche,
                    source=trend.source,
                    mentions_growth=trend.mentions_growth,
                    competition=trend.competition,
                    freshness_hours=trend.freshness_hours,
                    category=category,
                    intent=intent,
                )
            )
        return classified

    def _infer_category(self, text: str) -> str:
        if any(token in text for token in ["mistake", "myth", "stop"]):
            return "pain_point"
        if any(token in text for token in ["routine", "plan", "checklist", "breakdown"]):
            return "how_to"
        if any(token in text for token in ["before", "hidden", "hack", "trick"]):
            return "curiosity"
        return "general"

    def _infer_intent(self, text: str) -> str:
        if any(token in text for token in ["explained", "breakdown", "myth"]):
            return "educational"
        if any(token in text for token in ["routine", "plan", "minute"]):
            return "actionable"
        return "awareness"
