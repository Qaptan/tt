from pathlib import Path

from upload_engine.draft_or_publish_mode import DraftOrPublishMode
from upload_engine.upload_queue_manager import UploadQueueManager


def test_queue_lifecycle(tmp_path: Path) -> None:
    queue = UploadQueueManager(tmp_path / "queue.json")
    task = queue.enqueue(
        account_name="acct",
        video_path=tmp_path / "v.mp4",
        caption="c",
        hashtags=["#x"],
        mode=DraftOrPublishMode.DRAFT,
    )
    nxt = queue.next_pending()
    assert nxt is not None
    queue.mark_completed(task.id, post_id="post1")
    all_tasks = queue.all_tasks()
    assert all_tasks[0]["status"] == "completed"
