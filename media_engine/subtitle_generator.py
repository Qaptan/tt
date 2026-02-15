"""Subtitle generation using local Whisper or deterministic fallback."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.constants import WHISPER_BINARY


class SubtitleGenerator:
    """Create SRT subtitles from narration audio."""

    def __init__(self, logger: Any, timeout_seconds: int = 240) -> None:
        self.logger = logger
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        audio_path: Path,
        output_srt: Path,
        language: str,
        fallback_text: str,
        duration_seconds: int,
    ) -> Path:
        """Generate subtitles with Whisper if available; otherwise fallback."""
        output_srt.parent.mkdir(parents=True, exist_ok=True)

        if shutil.which(WHISPER_BINARY):
            produced = self._run_whisper(audio_path=audio_path, output_srt=output_srt, language=language)
            if produced:
                return produced

        self._write_fallback_srt(output_srt, fallback_text, duration_seconds)
        return output_srt

    def _run_whisper(self, audio_path: Path, output_srt: Path, language: str) -> Path | None:
        out_dir = output_srt.parent
        cmd = [
            WHISPER_BINARY,
            str(audio_path),
            "--task",
            "transcribe",
            "--language",
            language,
            "--output_format",
            "srt",
            "--output_dir",
            str(out_dir),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if result.returncode != 0:
                self.logger.warning("Whisper command failed", context={"stderr": result.stderr.strip()})
                return None
            generated = out_dir / f"{audio_path.stem}.srt"
            if generated.exists():
                generated.replace(output_srt)
                return output_srt
            return None
        except (OSError, subprocess.SubprocessError) as exc:
            self.logger.warning("Whisper unavailable", context={"error": str(exc)})
            return None

    def _write_fallback_srt(self, output_srt: Path, text: str, duration_seconds: int) -> None:
        words = text.split()
        if not words:
            words = ["..."]
        chunks = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]
        slot = max(1.0, duration_seconds / max(1, len(chunks)))

        lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            start = (idx - 1) * slot
            end = min(duration_seconds, idx * slot)
            lines.extend(
                [
                    str(idx),
                    f"{self._fmt(start)} --> {self._fmt(end)}",
                    chunk,
                    "",
                ]
            )
        output_srt.write_text("\n".join(lines), encoding="utf-8")

    def _fmt(self, seconds: float) -> str:
        millis = int((seconds - int(seconds)) * 1000)
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02}:{m:02}:{s:02},{millis:03}"
