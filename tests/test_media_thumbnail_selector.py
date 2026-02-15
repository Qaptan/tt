from pathlib import Path
from types import SimpleNamespace

import media_engine.thumbnail_selector as mod
from media_engine.thumbnail_selector import ThumbnailSelector


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_thumbnail_extract(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, capture_output, text, check):
        Path(cmd[-1]).write_bytes(b"png")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    selector = ThumbnailSelector(DummyLogger())
    out = selector.extract(tmp_path / "v.mp4", tmp_path / "t.png")
    assert out.name == "t.png"
