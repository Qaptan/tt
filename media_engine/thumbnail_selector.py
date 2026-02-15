"""Thumbnail extraction utility."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from core.constants import FFMPEG_BINARY


class ThumbnailSelector:
    """Select a representative frame as thumbnail."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def extract(self, video_path: Path, output_image: Path, timestamp_seconds: float = 1.0) -> Path:
        """Extract thumbnail frame from a rendered video."""
        output_image.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            FFMPEG_BINARY,
            "-y",
            "-ss",
            str(timestamp_seconds),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_image),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            self.logger.warning(
                "Thumbnail extraction failed",
                context={"stderr": result.stderr.strip(), "video": str(video_path)},
            )
        return output_image
