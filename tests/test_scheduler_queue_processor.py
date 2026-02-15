from pathlib import Path

from scheduler.queue_processor import QueueProcessor
from upload_engine.draft_or_publish_mode import DraftOrPublishMode
from upload_engine.upload_queue_manager import UploadQueueManager


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def test_queue_processor_success(tmp_path: Path) -> None:
    queue = UploadQueueManager(tmp_path / "q.json")
    queue.enqueue("acct", tmp_path / "v.mp4", "cap", ["#x"], DraftOrPublishMode.DRAFT)

    processor = QueueProcessor(DummyLogger())

    def upload_fn(task):
        return True, "post1", None

    result = processor.process(queue, upload_fn, attempts=3, backoff_seconds=0)
    assert result.succeeded == 1
    assert result.failed == 0
