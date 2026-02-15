"""Cron and interval scheduler helpers."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


class CronRunner:
    """Helper for cron setup and local interval loops."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def cron_line(self, project_root: Path, interval_minutes: int, command: str = "run") -> str:
        """Generate cron line for executing the engine."""
        python_bin = sys.executable
        minute_expr = f"*/{max(1, interval_minutes)}"
        return f"{minute_expr} * * * * cd {project_root} && {python_bin} main.py {command}"

    def install(self, cron_line: str) -> None:
        """Install cron line for current user."""
        existing = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
        current = existing.stdout if existing.returncode == 0 else ""
        if cron_line in current:
            self.logger.info("Cron line already installed")
            return

        merged = (current.strip() + "\n" + cron_line + "\n").strip() + "\n"
        proc = subprocess.run(["crontab", "-"], input=merged, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to install cron: {proc.stderr.strip()}")

    def run_interval_loop(self, interval_minutes: int, fn: Callable[[], None], cycles: int | None = None) -> None:
        """Run callback in a local loop, useful for foreground scheduling."""
        executed = 0
        while True:
            fn()
            executed += 1
            if cycles is not None and executed >= cycles:
                break
            time.sleep(max(1, interval_minutes * 60))
