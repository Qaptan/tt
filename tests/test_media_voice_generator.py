from pathlib import Path

from media_engine.voice_generator import VoiceGenerator


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_voice_fallback(monkeypatch, tmp_path: Path) -> None:
    gen = VoiceGenerator(DummyLogger())

    def fake_silence(path: Path, duration_seconds: int) -> None:
        path.write_bytes(b"RIFF")

    monkeypatch.setattr(gen, "_generate_silence", fake_silence)
    out = gen.synthesize("hello", tmp_path / "x.wav", voice_model_path=None, duration_seconds=3)
    assert out.exists()
