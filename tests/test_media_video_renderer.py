from pathlib import Path
from types import SimpleNamespace

import media_engine.video_renderer as mod
from media_engine.video_renderer import VideoRenderer


class DummyLogger:
    def error(self, *args, **kwargs):
        return None


def test_video_render_success(monkeypatch, tmp_path: Path) -> None:
    renderer = VideoRenderer(DummyLogger())

    def fake_run(cmd, capture_output, text, check):
        Path(cmd[-1]).write_bytes(b"mp4")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    out = renderer.render(
        output_video=tmp_path / "v.mp4",
        audio_path=tmp_path / "a.wav",
        subtitle_path=tmp_path / "s.srt",
        hook_text="hook",
        body_text="body",
        caption_text="cap",
        duration_seconds=9,
    )
    assert out.exists()
