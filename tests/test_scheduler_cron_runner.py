from pathlib import Path

from scheduler.cron_runner import CronRunner


class DummyLogger:
    def info(self, *args, **kwargs):
        return None


def test_cron_line_contains_main() -> None:
    line = CronRunner(DummyLogger()).cron_line(project_root=Path("/tmp/project"), interval_minutes=30)
    assert "main.py run" in line
