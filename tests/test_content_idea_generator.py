from content_engine.idea_generator import IdeaGenerator


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_fallback_ideas(monkeypatch) -> None:
    gen = IdeaGenerator(DummyLogger())
    monkeypatch.setattr(gen, "_ollama_available", lambda: False)
    ideas = gen.generate_ideas(
        account_name="acct",
        niche="fitness",
        model="llama3",
        scored_trends=[{"phrase": "high protein breakfast"}],
        format_name="story",
        count=2,
    )
    assert len(ideas) == 2
    assert ideas[0].format_name == "story"
