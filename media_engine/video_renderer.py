"""Video rendering pipeline using FFmpeg."""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.constants import FFMPEG_BINARY


class VideoRenderer:
    """Render 1080x1920 TikTok videos with animated text overlay."""

    def __init__(self, logger: Any, fps: int = 30, resolution: str = "1080x1920") -> None:
        self.logger = logger
        self.fps = fps
        self.resolution = resolution

    def render(
        self,
        output_video: Path,
        audio_path: Path,
        subtitle_path: Path,
        hook_text: str,
        body_text: str,
        caption_text: str,
        duration_seconds: int,
        background_music: Path | None = None,
        background_color: str | None = None,
    ) -> Path:
        """Render final MP4 output with H264 codec."""
        output_video.parent.mkdir(parents=True, exist_ok=True)

        if not shutil.which(FFMPEG_BINARY):
            raise RuntimeError("ffmpeg is required but not found on PATH")

        bg_color = background_color or random.choice(["0x101820", "0x1f2a44", "0x2d1e2f", "0x1f3b4d"])
        hook = self._escape(hook_text)
        body = self._escape(body_text)
        cap = self._escape(caption_text)

        filter_chain = (
            f"drawtext=text='{hook}':"
            "x=(w-text_w)/2:y=h*0.16:fontcolor=white:fontsize=if(lt(t,1.5),130-22*t,96):"
            "box=1:boxcolor=black@0.45:boxborderw=22:enable='between(t,0,1.5)',"
            f"drawtext=text='{body}':"
            "x=(w-text_w)/2:y=h*0.34:fontcolor=white:fontsize=56:"
            "box=1:boxcolor=black@0.4:boxborderw=18:enable='between(t,1.2,14)',"
            f"drawtext=text='{cap}':"
            "x=(w-text_w)/2:y=h*0.82:fontcolor=yellow:fontsize=46:"
            "box=1:boxcolor=black@0.35:boxborderw=14:enable='between(t,0,15)'"
        )

        cmd = [
            FFMPEG_BINARY,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={bg_color}:s={self.resolution}:r={self.fps}:d={duration_seconds}",
            "-i",
            str(audio_path),
        ]

        if background_music and background_music.exists():
            cmd += ["-i", str(background_music)]
            filter_complex = f"[1:a]volume=1.0[voice];[2:a]volume=0.15[music];[voice][music]amix=inputs=2:duration=first[a]"
            cmd += ["-filter_complex", filter_complex, "-map", "0:v", "-map", "[a]"]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]

        cmd += [
            "-vf",
            filter_chain,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(self.fps),
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_video),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            self.logger.error(
                "Video rendering failed",
                context={"stderr": result.stderr.strip(), "video": str(output_video)},
            )
            raise RuntimeError("Video rendering failed")

        if subtitle_path.exists():
            self._burn_subtitles(output_video, subtitle_path)

        return output_video

    def _burn_subtitles(self, video_path: Path, subtitle_path: Path) -> None:
        temp_path = video_path.with_name(f"{video_path.stem}_subtitled{video_path.suffix}")
        cmd = [
            FFMPEG_BINARY,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles={self._escape_path(subtitle_path)}:force_style='Fontsize=20,PrimaryColour=&H00FFFFFF,Outline=1'",
            "-c:a",
            "copy",
            str(temp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and temp_path.exists():
            temp_path.replace(video_path)

    def _escape(self, text: str) -> str:
        return text.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")

    def _escape_path(self, path: Path) -> str:
        return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
