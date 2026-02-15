from pathlib import Path

from upload_engine.draft_or_publish_mode import DraftOrPublishMode
from upload_engine.tiktok_uploader import TikTokUploader


class DummyLogger:
    def info(self, *args, **kwargs):
        return None


def test_mock_upload_returns_post_id(tmp_path: Path) -> None:
    uploader = TikTokUploader(DummyLogger(), upload_url="https://www.tiktok.com/upload", mock_mode=True)
    result = uploader.upload(
        video_path=tmp_path / "x.mp4",
        caption="caption",
        hashtags=["#a"],
        mode=DraftOrPublishMode.DRAFT,
        session_state_path=tmp_path / "session.json",
    )
    assert result.success is True
    assert result.post_id is not None
