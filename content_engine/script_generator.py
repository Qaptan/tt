"""TikTok short script generation."""

from __future__ import annotations

from dataclasses import dataclass

from content_engine.idea_generator import ContentIdea


@dataclass
class VideoScript:
    """Generated short-form script."""

    hook: str
    body: str
    cta: str
    full_text: str
    duration_seconds: int


class ScriptGenerator:
    """Generate concise scripts that fit 9-15 second clips."""

    def generate(self, idea: ContentIdea, hook_style: str, duration_seconds: int) -> VideoScript:
        """Build a script from an idea."""
        hook_templates = {
            "question": f"Still struggling with {idea.trend_phrase}?",
            "shock": f"Most people fail {idea.trend_phrase} for one reason.",
            "contrarian": f"Ignore the usual advice about {idea.trend_phrase}.",
            "promise": f"In {duration_seconds} seconds, fix {idea.trend_phrase}.",
        }
        hook = hook_templates.get(hook_style, hook_templates["question"])
        body = (
            f"Start with this: {idea.angle}. "
            f"Then apply one clear action today and keep it consistent."
        )
        cta = "Comment 'guide' and follow for the next breakdown."
        full_text = f"{hook} {body} {cta}"
        return VideoScript(hook=hook, body=body, cta=cta, full_text=full_text, duration_seconds=duration_seconds)
