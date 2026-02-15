"""Strategy decision model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class StrategyDecision:
    """Selected strategy for a new video."""

    format_name: str
    hook_style: str
    duration_seconds: int
    caption_style: str
    publish_hour: int


class DecisionModel:
    """Sample next content strategy from weighted profile."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def decide(
        self,
        strategy_model: dict[str, Any],
        top_trend_category: str | None,
        now: datetime | None = None,
    ) -> StrategyDecision:
        """Return next strategy decision using weighted or Thompson-based sampling."""
        format_weights = dict(strategy_model.get("format_weights", {}))
        hook_weights = dict(strategy_model.get("hook_weights", {}))
        caption_weights = dict(strategy_model.get("caption_weights", {}))
        duration_weights = {int(k): float(v) for k, v in strategy_model.get("duration_weights", {}).items()}
        hour_weights = {int(k): float(v) for k, v in strategy_model.get("publish_hour_weights", {}).items()}
        bandit_state = strategy_model.get("bandit_state", {})

        self._apply_trend_bias(format_weights, top_trend_category)

        format_name = self._thompson_or_weighted_choice(
            bucket_weights=format_weights,
            bucket_state=bandit_state.get("format", {}),
            default="story",
        )
        hook_style = self._thompson_or_weighted_choice(
            bucket_weights=hook_weights,
            bucket_state=bandit_state.get("hook", {}),
            default="question",
        )
        caption_style = self._thompson_or_weighted_choice(
            bucket_weights=caption_weights,
            bucket_state=bandit_state.get("caption", {}),
            default="direct",
        )
        duration_seconds = int(
            self._thompson_or_weighted_choice(
                bucket_weights=duration_weights,
                bucket_state=bandit_state.get("duration", {}),
                default=12,
            )
        )

        if now is not None and now.hour in hour_weights:
            hour_weights[now.hour] *= 1.05
        publish_hour = int(
            self._thompson_or_weighted_choice(
                bucket_weights=hour_weights,
                bucket_state=bandit_state.get("publish_hour", {}),
                default=12,
            )
        )

        return StrategyDecision(
            format_name=format_name,
            hook_style=hook_style,
            duration_seconds=duration_seconds,
            caption_style=caption_style,
            publish_hour=publish_hour,
        )

    def _apply_trend_bias(self, format_weights: dict[str, float], top_category: str | None) -> None:
        if not top_category or not format_weights:
            return
        if top_category == "how_to":
            for key in ["tutorial", "listicle"]:
                if key in format_weights:
                    format_weights[key] *= 1.15
        elif top_category == "curiosity":
            for key in ["story", "challenge"]:
                if key in format_weights:
                    format_weights[key] *= 1.15
        elif top_category == "pain_point":
            for key in ["listicle", "story"]:
                if key in format_weights:
                    format_weights[key] *= 1.1

    def _weighted_choice(self, weights: dict[Any, float], default: Any) -> Any:
        if not weights:
            return default
        keys = list(weights.keys())
        vals = [max(0.0001, float(weights[k])) for k in keys]
        total = sum(vals)
        threshold = self._random.uniform(0.0, total)
        cumulative = 0.0
        for key, weight in zip(keys, vals):
            cumulative += weight
            if cumulative >= threshold:
                return key
        return keys[-1]

    def _thompson_or_weighted_choice(
        self,
        bucket_weights: dict[Any, float],
        bucket_state: dict[str, Any],
        default: Any,
    ) -> Any:
        """Select value by Thompson Sampling when posterior state exists."""
        if not bucket_weights:
            return default
        if not bucket_state:
            return self._weighted_choice(bucket_weights, default=default)

        sampled_scores: dict[Any, float] = {}
        for key, weight in bucket_weights.items():
            state_key = str(key)
            raw_state = bucket_state.get(state_key, {})
            alpha = max(0.1, float(raw_state.get("alpha", 1.0)))
            beta = max(0.1, float(raw_state.get("beta", 1.0)))
            posterior_sample = self._random.betavariate(alpha, beta)
            sampled_scores[key] = posterior_sample * max(0.0001, float(weight))

        if not sampled_scores:
            return self._weighted_choice(bucket_weights, default=default)
        return max(sampled_scores.items(), key=lambda item: item[1])[0]
