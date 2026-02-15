"""Content idea generation via local Ollama model."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from core.constants import OLLAMA_BINARY


@dataclass
class ContentIdea:
    """Single generated content idea."""

    title: str
    angle: str
    target_emotion: str
    trend_phrase: str
    format_name: str


class IdeaGenerator:
    """Generate short-form content ideas using local LLM."""

    def __init__(self, logger: Any, timeout_seconds: int = 90) -> None:
        self.logger = logger
        self.timeout_seconds = timeout_seconds

    def generate_ideas(
        self,
        account_name: str,
        niche: str,
        model: str,
        scored_trends: list[dict[str, Any]],
        format_name: str,
        count: int = 3,
    ) -> list[ContentIdea]:
        """Generate content ideas from top trends."""
        top_trends = [item["phrase"] for item in scored_trends[:max(1, count)]]
        if self._ollama_available():
            ideas = self._generate_with_ollama(
                account_name=account_name,
                niche=niche,
                model=model,
                top_trends=top_trends,
                format_name=format_name,
                count=count,
            )
            if ideas:
                return ideas

        return self._fallback_ideas(niche=niche, top_trends=top_trends, format_name=format_name, count=count)

    def _generate_with_ollama(
        self,
        account_name: str,
        niche: str,
        model: str,
        top_trends: list[str],
        format_name: str,
        count: int,
    ) -> list[ContentIdea]:
        prompt = (
            f"You are a TikTok strategist for account '{account_name}' in niche '{niche}'. "
            f"Generate {count} ideas in format '{format_name}'. "
            "Output each idea on one line with this pipe format: "
            "Title|Angle|Emotion|TrendPhrase. "
            f"Trends: {', '.join(top_trends)}"
        )
        try:
            result = subprocess.run(
                [OLLAMA_BINARY, "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if result.returncode != 0:
                self.logger.warning(
                    "Ollama command failed; using fallback ideas",
                    context={"stderr": result.stderr.strip()},
                )
                return []

            ideas: list[ContentIdea] = []
            for line in result.stdout.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 4:
                    continue
                ideas.append(
                    ContentIdea(
                        title=parts[0],
                        angle=parts[1],
                        target_emotion=parts[2],
                        trend_phrase=parts[3],
                        format_name=format_name,
                    )
                )
            return ideas[:count]
        except (subprocess.SubprocessError, OSError) as exc:
            self.logger.warning(
                "Ollama unavailable during generation; using fallback",
                context={"error": str(exc)},
            )
            return []

    def _fallback_ideas(
        self,
        niche: str,
        top_trends: list[str],
        format_name: str,
        count: int,
    ) -> list[ContentIdea]:
        ideas: list[ContentIdea] = []
        trends = top_trends or [f"{niche} quick win"]
        emotions = ["curiosity", "motivation", "confidence", "urgency"]

        for i in range(count):
            trend = trends[i % len(trends)]
            emotion = emotions[i % len(emotions)]
            ideas.append(
                ContentIdea(
                    title=f"{trend.title()} in {i + 1} steps",
                    angle=f"Actionable {format_name} take on {trend}",
                    target_emotion=emotion,
                    trend_phrase=trend,
                    format_name=format_name,
                )
            )
        return ideas

    def _ollama_available(self) -> bool:
        return shutil.which(OLLAMA_BINARY) is not None
