"""Persistent strategy weight model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FormatScoreModel:
    """Load, update, and save strategy weights."""

    def __init__(self, model_path: Path, rotation_floor: float = 0.05) -> None:
        self.model_path = model_path
        self.rotation_floor = rotation_floor
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def load_or_init(self, strategy_config: dict[str, Any]) -> dict[str, Any]:
        """Load existing model, or initialize from config."""
        if self.model_path.exists():
            model = json.loads(self.model_path.read_text(encoding="utf-8"))
            changed = self._ensure_schema(model, strategy_config)
            if changed:
                self.save(model)
            return model

        model = {
            "format_weights": dict(strategy_config.get("initial_format_weights", {})),
            "hook_weights": dict(strategy_config.get("initial_hook_weights", {})),
            "caption_weights": dict(strategy_config.get("initial_caption_weights", {})),
            "duration_weights": {
                str(k): float(v)
                for k, v in dict(strategy_config.get("initial_duration_weights", {})).items()
            },
            "publish_hour_weights": {str(hour): 1.0 for hour in range(24)},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._ensure_schema(model, strategy_config)
        self.save(model)
        return model

    def save(self, model: dict[str, Any]) -> None:
        """Persist strategy model."""
        model["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    def update(
        self,
        model: dict[str, Any],
        winners: dict[str, list[Any]],
        losers: dict[str, list[Any]],
        learning_rate: float,
    ) -> dict[str, Any]:
        """Update strategy weights from winners/losers patterns."""
        self._ensure_schema(model, {})
        self._adjust(model["format_weights"], winners.get("formats", []), losers.get("formats", []), learning_rate)
        self._adjust(model["hook_weights"], winners.get("hooks", []), losers.get("hooks", []), learning_rate)
        self._adjust(model["caption_weights"], winners.get("caption_styles", []), losers.get("caption_styles", []), learning_rate)
        self._adjust(model["duration_weights"], winners.get("durations", []), losers.get("durations", []), learning_rate)
        self._adjust(model["publish_hour_weights"], winners.get("publish_hours", []), losers.get("publish_hours", []), learning_rate)
        self._update_bandit_state(model, winners=winners, losers=losers, learning_rate=learning_rate)

        self._normalize_all(model)
        return model

    def _adjust(
        self,
        weights: dict[str, float],
        winners: list[Any],
        losers: list[Any],
        learning_rate: float,
    ) -> None:
        for winner in winners:
            key = str(winner)
            if key in weights:
                weights[key] = float(weights[key]) * (1.0 + learning_rate)

        for loser in losers:
            key = str(loser)
            if key in weights:
                weights[key] = max(self.rotation_floor, float(weights[key]) * (1.0 - learning_rate))

    def _normalize_all(self, model: dict[str, Any]) -> None:
        for bucket in [
            model.get("format_weights", {}),
            model.get("hook_weights", {}),
            model.get("caption_weights", {}),
            model.get("duration_weights", {}),
            model.get("publish_hour_weights", {}),
        ]:
            self._normalize(bucket)

    def _normalize(self, weights: dict[str, float]) -> None:
        if not weights:
            return
        total = sum(max(self.rotation_floor, float(v)) for v in weights.values())
        if total <= 0:
            even = 1.0 / len(weights)
            for key in weights:
                weights[key] = even
            return
        for key in weights:
            weights[key] = max(self.rotation_floor, float(weights[key])) / total

    def _ensure_schema(self, model: dict[str, Any], strategy_config: dict[str, Any]) -> bool:
        """Ensure model keeps backward-compatible fields and Thompson state."""
        changed = False

        defaults = {
            "format_weights": dict(strategy_config.get("initial_format_weights", {})),
            "hook_weights": dict(strategy_config.get("initial_hook_weights", {})),
            "caption_weights": dict(strategy_config.get("initial_caption_weights", {})),
            "duration_weights": {
                str(k): float(v)
                for k, v in dict(strategy_config.get("initial_duration_weights", {})).items()
            },
        }
        for field, fallback in defaults.items():
            if field not in model:
                model[field] = fallback
                changed = True

        if "publish_hour_weights" not in model:
            model["publish_hour_weights"] = {str(hour): 1.0 for hour in range(24)}
            changed = True

        bandit = model.setdefault("bandit_state", {})
        if not isinstance(bandit, dict):
            model["bandit_state"] = {}
            bandit = model["bandit_state"]
            changed = True

        bucket_specs = {
            "format": model.get("format_weights", {}).keys(),
            "hook": model.get("hook_weights", {}).keys(),
            "caption": model.get("caption_weights", {}).keys(),
            "duration": model.get("duration_weights", {}).keys(),
            "publish_hour": model.get("publish_hour_weights", {}).keys(),
        }
        for bucket_name, keys in bucket_specs.items():
            bucket = bandit.setdefault(bucket_name, {})
            if not isinstance(bucket, dict):
                bandit[bucket_name] = {}
                bucket = bandit[bucket_name]
                changed = True
            for key in keys:
                skey = str(key)
                if skey not in bucket:
                    bucket[skey] = {"alpha": 1.0, "beta": 1.0}
                    changed = True
                    continue
                alpha = float(bucket[skey].get("alpha", 1.0))
                beta = float(bucket[skey].get("beta", 1.0))
                if alpha <= 0 or beta <= 0:
                    bucket[skey] = {"alpha": max(0.1, alpha), "beta": max(0.1, beta)}
                    changed = True

        return changed

    def _update_bandit_state(
        self,
        model: dict[str, Any],
        winners: dict[str, list[Any]],
        losers: dict[str, list[Any]],
        learning_rate: float,
    ) -> None:
        delta = max(0.1, float(learning_rate))
        bandit = model.get("bandit_state", {})

        self._apply_bandit_delta(bandit.get("format", {}), winners.get("formats", []), "alpha", delta)
        self._apply_bandit_delta(bandit.get("format", {}), losers.get("formats", []), "beta", delta)

        self._apply_bandit_delta(bandit.get("hook", {}), winners.get("hooks", []), "alpha", delta)
        self._apply_bandit_delta(bandit.get("hook", {}), losers.get("hooks", []), "beta", delta)

        self._apply_bandit_delta(bandit.get("caption", {}), winners.get("caption_styles", []), "alpha", delta)
        self._apply_bandit_delta(bandit.get("caption", {}), losers.get("caption_styles", []), "beta", delta)

        self._apply_bandit_delta(bandit.get("duration", {}), winners.get("durations", []), "alpha", delta)
        self._apply_bandit_delta(bandit.get("duration", {}), losers.get("durations", []), "beta", delta)

        self._apply_bandit_delta(bandit.get("publish_hour", {}), winners.get("publish_hours", []), "alpha", delta)
        self._apply_bandit_delta(bandit.get("publish_hour", {}), losers.get("publish_hours", []), "beta", delta)

    def _apply_bandit_delta(
        self,
        bucket: dict[str, Any],
        keys: list[Any],
        parameter: str,
        delta: float,
    ) -> None:
        if not isinstance(bucket, dict):
            return
        for key in keys:
            skey = str(key)
            if skey not in bucket:
                bucket[skey] = {"alpha": 1.0, "beta": 1.0}
            current = float(bucket[skey].get(parameter, 1.0))
            bucket[skey][parameter] = max(0.1, current + delta)
