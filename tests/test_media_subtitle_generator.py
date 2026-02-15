from pathlib import Path

from media_engine.subtitle_generator import SubtitleGenerator


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_subtitle_fallback(tmp_path: Path) -> None:
    gen = SubtitleGenerator(DummyLogger())
    srt = gen.generate(
        audio_path=tmp_path / "audio.wav",
        output_srt=tmp_path / "out.srt",
        language="en",
        fallback_text="one two three four five six",
        duration_seconds=6,
    )
    assert srt.exists()
    assert "-->" in srt.read_text(encoding="utf-8")
