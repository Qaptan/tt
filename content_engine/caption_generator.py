"""Caption generation module."""

from __future__ import annotations

from content_engine.idea_generator import ContentIdea


class CaptionGenerator:
    """Generate style-specific TikTok captions."""

    def generate(self, idea: ContentIdea, caption_style: str, include_cta: bool = True) -> str:
        """Return a short caption with chosen style."""
        style_prefix = {
            "direct": "Do this today:",
            "curiosity": "Most people miss this:",
            "urgency": "Try this before your next scroll:",
            "community": "Who else is working on this:",
        }
        prefix = style_prefix.get(caption_style, "Try this:")
        caption = f"{prefix} {idea.title}"
        if include_cta:
            caption += " | Save this and share it."
        return caption
