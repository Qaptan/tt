from content_engine.idea_generator import ContentIdea
from content_engine.script_generator import ScriptGenerator


def test_script_generation() -> None:
    idea = ContentIdea("Title", "Angle", "curiosity", "trend", "story")
    script = ScriptGenerator().generate(idea, hook_style="question", duration_seconds=12)
    assert "trend" in script.hook.lower()
    assert script.duration_seconds == 12
