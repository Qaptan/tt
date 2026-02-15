import json
from pathlib import Path

from upload_engine.draft_or_publish_mode import DraftOrPublishMode
from upload_engine.tiktok_uploader import TikTokUploader


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def test_official_api_upload_refreshes_token_and_uploads(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "session_state.json"
    session_path.write_text(
        json.dumps(
            {
                "access_token": "expired_token",
                "refresh_token": "refresh_123",
                "expires_at": "2000-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake-mp4")

    uploader = TikTokUploader(
        logger=DummyLogger(),
        upload_url="https://www.tiktok.com/upload",
        mock_mode=False,
        api_base_url="https://open.tiktokapis.com",
        client_key="client_key",
        client_secret="client_secret",
    )

    calls: list[tuple[str, str]] = []

    def fake_request_with_retry(method: str, url: str, headers: dict[str, str], data: bytes | None, expect_json: bool):
        del data
        del expect_json
        calls.append((method, url))

        if url.endswith("/v2/oauth/token/"):
            return {
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
                "expires_in": 3600,
            }
        if url.endswith("/v2/post/publish/inbox/video/init/"):
            assert headers["Authorization"] == "Bearer new_access_token"
            return {
                "data": {
                    "upload_url": "https://upload.example.local/obj/abc",
                    "publish_id": "publish_999",
                }
            }
        if url == "https://upload.example.local/obj/abc":
            return {}

        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(uploader, "_request_with_retry", fake_request_with_retry)

    result = uploader.upload(
        video_path=video_path,
        caption="hello",
        hashtags=["#tag1"],
        mode=DraftOrPublishMode.DRAFT,
        session_state_path=session_path,
    )

    assert result.success is True
    assert result.post_id == "publish_999"
    assert any(url.endswith("/v2/oauth/token/") for _, url in calls)
    assert any(url.endswith("/v2/post/publish/inbox/video/init/") for _, url in calls)

    refreshed = json.loads(session_path.read_text(encoding="utf-8"))
    assert refreshed["access_token"] == "new_access_token"
    assert refreshed["refresh_token"] == "new_refresh_token"
