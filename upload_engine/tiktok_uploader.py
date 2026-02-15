"""TikTok uploader using the official TikTok Content Posting API."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse, request

from upload_engine.draft_or_publish_mode import DraftOrPublishMode


@dataclass
class UploadResult:
    """Result of upload action."""

    success: bool
    post_id: str | None
    error: str | None = None


class TikTokApiError(RuntimeError):
    """API-level error with retry metadata."""

    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class TikTokUploader:
    """Upload videos through TikTok Content Posting API with OAuth token refresh."""

    def __init__(
        self,
        logger: Any,
        upload_url: str,
        headless: bool = True,
        mock_mode: bool = True,
        api_base_url: str | None = None,
        client_key: str | None = None,
        client_secret: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.logger = logger
        # `upload_url` and `headless` are kept for backward compatibility with old configs.
        self.upload_url = upload_url
        self.headless = headless
        self.mock_mode = mock_mode
        self.api_base_url = (api_base_url or os.getenv("TIKTOK_API_BASE_URL", "https://open.tiktokapis.com")).rstrip("/")
        self.client_key = client_key or os.getenv("TIKTOK_CLIENT_KEY")
        self.client_secret = client_secret or os.getenv("TIKTOK_CLIENT_SECRET")
        self.timeout_seconds = max(5, int(timeout_seconds))
        self.max_api_attempts = 3

    def upload(
        self,
        video_path: Path,
        caption: str,
        hashtags: list[str],
        mode: DraftOrPublishMode,
        session_state_path: Path,
        post_wait_seconds: int = 12,
    ) -> UploadResult:
        """Upload a video in draft or direct mode."""
        if self.mock_mode:
            post_id = f"mock_{uuid.uuid4().hex[:12]}"
            self.logger.info("Mock upload complete", context={"post_id": post_id, "video": str(video_path)})
            return UploadResult(success=True, post_id=post_id)

        if not video_path.exists():
            return UploadResult(success=False, post_id=None, error=f"Video not found: {video_path}")

        try:
            token_state = self._load_token_state(session_state_path)
            access_token = self._ensure_access_token(token_state, session_state_path)

            full_caption = self._build_caption(caption, hashtags)
            init_path = (
                "/v2/post/publish/inbox/video/init/"
                if mode == DraftOrPublishMode.DRAFT
                else "/v2/post/publish/video/init/"
            )
            init_payload = self._build_init_payload(video_path=video_path, title=full_caption)
            init_response = self._api_request(
                method="POST",
                path=init_path,
                access_token=access_token,
                json_payload=init_payload,
            )
            upload_url, publish_id = self._extract_upload_info(init_response)
            if not upload_url:
                return UploadResult(success=False, post_id=None, error="TikTok API did not return upload URL")

            self._upload_binary(upload_url=upload_url, video_path=video_path)

            post_id = publish_id or f"tt_{uuid.uuid4().hex[:12]}"
            self.logger.info(
                "Official API upload complete",
                context={
                    "post_id": post_id,
                    "mode": mode.value,
                    "video": str(video_path),
                    "session_file": str(session_state_path),
                },
            )
            return UploadResult(success=True, post_id=post_id)
        except TikTokApiError as exc:
            return UploadResult(success=False, post_id=None, error=f"TikTok API error: {exc}")
        except Exception as exc:  # noqa: BLE001
            return UploadResult(success=False, post_id=None, error=f"Upload failed: {exc}")

    def _build_caption(self, caption: str, hashtags: list[str]) -> str:
        tags = " ".join(hashtags)
        return f"{caption}\n{tags}".strip()[:2200]

    def _build_init_payload(self, video_path: Path, title: str) -> dict[str, Any]:
        video_size = int(video_path.stat().st_size)
        return {
            "post_info": {
                "title": title,
                "privacy_level": "PUBLIC_TO_EVERYONE",
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": video_size,
                "total_chunk_count": 1,
            },
        }

    def _load_token_state(self, session_state_path: Path) -> dict[str, Any]:
        if not session_state_path.exists():
            raise TikTokApiError(
                f"Token state not found: {session_state_path}. Save OAuth tokens for this account first."
            )
        payload = json.loads(session_state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TikTokApiError(f"Invalid token state file: {session_state_path}")
        return payload

    def _ensure_access_token(self, token_state: dict[str, Any], session_state_path: Path) -> str:
        access_token = str(token_state.get("access_token", "")).strip()
        refresh_token = str(token_state.get("refresh_token", "")).strip()

        if access_token and not self._is_token_expired(token_state):
            return access_token

        if not refresh_token:
            raise TikTokApiError("Access token expired and refresh_token is missing")
        if not self.client_key or not self.client_secret:
            raise TikTokApiError(
                "Missing TikTok OAuth credentials. Set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET."
            )

        refreshed = self._refresh_access_token(refresh_token)
        token_state["access_token"] = str(refreshed.get("access_token", "")).strip()
        token_state["refresh_token"] = str(refreshed.get("refresh_token", refresh_token)).strip()
        expires_in = int(refreshed.get("expires_in", 3600))
        token_state["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
        if refreshed.get("open_id"):
            token_state["open_id"] = refreshed["open_id"]

        self._persist_token_state(session_state_path, token_state)
        if not token_state["access_token"]:
            raise TikTokApiError("Token refresh completed without access_token")
        return token_state["access_token"]

    def _is_token_expired(self, token_state: dict[str, Any]) -> bool:
        expires_at = token_state.get("expires_at")
        if not expires_at:
            return False
        try:
            expiry = datetime.fromisoformat(str(expires_at))
        except ValueError:
            return False
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        # refresh a bit early to avoid near-expiry failures
        return datetime.now(timezone.utc) >= (expiry - timedelta(seconds=60))

    def _refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        token_url = f"{self.api_base_url}/v2/oauth/token/"
        body = parse.urlencode(
            {
                "client_key": self.client_key or "",
                "client_secret": self.client_secret or "",
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = self._request_with_retry(
            method="POST",
            url=token_url,
            headers=headers,
            data=body,
            expect_json=True,
        )
        data = response.get("data", response)
        if not isinstance(data, dict):
            raise TikTokApiError("Unexpected OAuth token response payload")
        return data

    def _persist_token_state(self, session_state_path: Path, token_state: dict[str, Any]) -> None:
        session_state_path.parent.mkdir(parents=True, exist_ok=True)
        session_state_path.write_text(json.dumps(token_state, indent=2), encoding="utf-8")
        try:
            os.chmod(session_state_path, 0o600)
        except OSError:
            pass

    def _api_request(
        self,
        method: str,
        path: str,
        access_token: str,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.api_base_url}{path}"
        data = json.dumps(json_payload or {}).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        response = self._request_with_retry(
            method=method,
            url=url,
            headers=headers,
            data=data,
            expect_json=True,
        )
        return response

    def _extract_upload_info(self, init_response: dict[str, Any]) -> tuple[str | None, str | None]:
        data = init_response.get("data", init_response)
        if not isinstance(data, dict):
            return None, None
        upload_url = data.get("upload_url")
        if not upload_url:
            url_list = data.get("upload_urls") or data.get("upload_url_list") or []
            if isinstance(url_list, list) and url_list:
                upload_url = url_list[0]
        publish_id = data.get("publish_id") or data.get("post_id") or data.get("video_id")
        return (str(upload_url) if upload_url else None, str(publish_id) if publish_id else None)

    def _upload_binary(self, upload_url: str, video_path: Path) -> None:
        payload = video_path.read_bytes()
        headers = {
            "Content-Type": "video/mp4",
            "Content-Length": str(len(payload)),
        }
        try:
            self._request_with_retry(
                method="PUT",
                url=upload_url,
                headers=headers,
                data=payload,
                expect_json=False,
            )
        except TikTokApiError:
            # Some upload URLs accept POST rather than PUT.
            self._request_with_retry(
                method="POST",
                url=upload_url,
                headers=headers,
                data=payload,
                expect_json=False,
            )

    def _request_with_retry(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: bytes | None,
        expect_json: bool,
    ) -> dict[str, Any]:
        for attempt in range(1, self.max_api_attempts + 1):
            try:
                return self._http_request(method=method, url=url, headers=headers, data=data, expect_json=expect_json)
            except TikTokApiError as exc:
                if not exc.retryable or attempt >= self.max_api_attempts:
                    raise
                sleep_for = 2 ** (attempt - 1)
                self.logger.warning(
                    "TikTok API request failed; retrying",
                    context={
                        "url": url,
                        "attempt": attempt,
                        "max_attempts": self.max_api_attempts,
                        "error": str(exc),
                        "sleep_seconds": sleep_for,
                    },
                )
                time.sleep(sleep_for)
        raise TikTokApiError("Request retry loop exhausted")

    def _http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: bytes | None,
        expect_json: bool,
    ) -> dict[str, Any]:
        req = request.Request(url=url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
                if not expect_json:
                    return {}
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {429, 500, 502, 503, 504}
            raise TikTokApiError(
                f"HTTP {exc.code}: {body[:500]}",
                status_code=exc.code,
                retryable=retryable,
            ) from exc
        except urllib_error.URLError as exc:
            raise TikTokApiError(f"Network error: {exc.reason}", retryable=True) from exc
