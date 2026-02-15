"""Voice generation using local Piper TTS."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.constants import FFMPEG_BINARY, PIPER_BINARY


class VoiceGenerator:
    """Generate narration tracks from script text."""

    def __init__(self, logger: Any, timeout_seconds: int = 120) -> None:
        self.logger = logger
        self.timeout_seconds = timeout_seconds

    def synthesize(
        self,
        text: str,
        output_wav: Path,
        voice_model_path: str | None,
        duration_seconds: int,
    ) -> Path:
        """Synthesize voice with Piper, fallback to silent track."""
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        if voice_model_path and shutil.which(PIPER_BINARY):
            try:
                cmd = [
                    PIPER_BINARY,
                    "--model",
                    voice_model_path,
                    "--output_file",
                    str(output_wav),
                ]
                result = subprocess.run(
                    cmd,
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                if result.returncode == 0 and output_wav.exists():
                    return output_wav
                self.logger.warning(
                    "Piper generation failed; falling back to silent audio",
                    context={"stderr": result.stderr.strip()},
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self.logger.warning(
                    "Piper unavailable; falling back to silent audio",
                    context={"error": str(exc)},
                )

        self._generate_silence(output_wav, duration_seconds)
        return output_wav

    def _generate_silence(self, output_wav: Path, duration_seconds: int) -> None:
        cmd = [
            FFMPEG_BINARY,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            str(max(1, duration_seconds)),
            str(output_wav),
        ]
        subprocess.run(cmd, capture_output=True, check=False)
