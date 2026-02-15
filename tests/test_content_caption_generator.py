from content_engine.caption_generator import CaptionGenerator
from content_engine.idea_generator import ContentIdea


def test_caption_style() -> None:
    idea = ContentIdea("Test title", "angle", "emotion", "trend", "story")
    caption = CaptionGenerator().generate(idea, caption_style="urgency")
    assert caption.startswith("Try this before")
