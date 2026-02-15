from pathlib import Path

from analytics_engine.metrics_collector import MetricsCollector


class DummyLogger:
    def info(self, *args, **kwargs):
        return None


def test_metrics_pending_to_collected(tmp_path: Path) -> None:
    collector = MetricsCollector(tmp_path / "metrics.jsonl", DummyLogger(), seed=1)
    collector.record_pending(
        post_id="p1",
        account_name="acct",
        metadata={
            "format_name": "story",
            "hook_style": "question",
            "duration_seconds": 9,
            "caption_style": "direct",
            "publish_hour": 12,
        },
    )
    collected = collector.collect(min_age_minutes=0, mock_mode=True)
    assert len(collected) == 1
    assert collected[0].status == "collected"
