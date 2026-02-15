"""Helper utilities for strategy weight boosting."""

from __future__ import annotations

from typing import Any


class FormatMultiplier:
    """Apply multiplicative updates to strategy profiles."""

    def boost(self, weights: dict[str, float], key: str, multiplier: float = 1.1) -> dict[str, float]:
        """Boost a specific key weight and return normalized distribution."""
        if key in weights:
            weights[key] *= multiplier
        return self._normalize(weights)

    def dampen(self, weights: dict[str, float], key: str, multiplier: float = 0.9, floor: float = 0.05) -> dict[str, float]:
        """Reduce a specific key weight and return normalized distribution."""
        if key in weights:
            weights[key] = max(floor, weights[key] * multiplier)
        return self._normalize(weights)

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        if not weights:
            return weights
        total = sum(weights.values())
        if total <= 0:
            even = 1.0 / len(weights)
            return {k: even for k in weights}
        return {k: v / total for k, v in weights.items()}

    def rotate(self, weights: dict[str, float], min_share: float = 0.05) -> dict[str, float]:
        """Guarantee exploration floor for each style."""
        if not weights:
            return weights
        adjusted = {k: max(min_share, v) for k, v in weights.items()}
        return self._normalize(adjusted)

    def apply_many(self, weights: dict[str, float], boosts: list[str], dampens: list[str]) -> dict[str, float]:
        """Apply bulk winner/loser updates."""
        for key in boosts:
            if key in weights:
                weights[key] *= 1.1
        for key in dampens:
            if key in weights:
                weights[key] *= 0.9
        return self._normalize(weights)
