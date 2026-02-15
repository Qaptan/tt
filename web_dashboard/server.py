"""HTTP dashboard server for controlling the autonomous growth engine."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml


@dataclass
class DashboardState:
    """Runtime state shared by dashboard request handlers."""

    engine: Any
    config_path: Path
    started_at: float
    lock: threading.Lock


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve static UI and JSON API endpoints."""

    state: DashboardState
    static_root: Path

    server_version = "AutonomousDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            route = parsed.path

            if route == "/api/health":
                self._handle_health()
                return
            if route == "/api/accounts":
                self._handle_accounts()
                return
            if route == "/api/stats":
                params = parse_qs(parsed.query)
                account = params.get("account", [None])[0]
                self._handle_stats(account_name=account)
                return
            if route == "/api/config":
                self._handle_config()
                return
            if route == "/api/logs":
                params = parse_qs(parsed.query)
                raw_lines = params.get("lines", ["80"])[0]
                lines = self._safe_int(raw_lines, default=80, minimum=10, maximum=500)
                self._handle_logs(lines=lines)
                return
            if route == "/" or route == "/index.html":
                self._serve_static("index.html")
                return
            if route.startswith("/assets/"):
                rel = route.removeprefix("/assets/")
                self._serve_static(rel)
                return

            self._send_json({"status": "error", "message": "Bulunamadi"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"status": "error", "message": f"Sunucu hatasi: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            route = parsed.path
            body = self._read_json_body()
            account = body.get("account")

            if route == "/api/init":
                with self.state.lock:
                    self.state.engine.init()
                self._send_json({"status": "ok", "message": "Baslatma tamamlandi"})
                return

            if route == "/api/accounts/upsert":
                saved = self._handle_account_upsert(body)
                self._send_json({"status": "ok", "message": "Hesap kaydedildi", "data": saved})
                return

            if route == "/api/accounts/delete":
                deleted_name = self._handle_account_delete(body)
                self._send_json({"status": "ok", "message": "Hesap silindi", "data": {"name": deleted_name}})
                return

            if route == "/api/run":
                with self.state.lock:
                    result = self.state.engine.run(account_name=account)
                self._send_json({"status": "ok", "data": [asdict(row) for row in result]})
                return

            if route == "/api/analyze":
                with self.state.lock:
                    analysis = self.state.engine.analyze(account_name=account)
                self._send_json({"status": "ok", "data": analysis})
                return

            if route == "/api/retrain":
                with self.state.lock:
                    update = self.state.engine.retrain(account_name=account)
                self._send_json({"status": "ok", "data": update})
                return

            if route == "/api/schedule":
                cycles = self._safe_int(body.get("cycles", 1), default=1, minimum=1, maximum=500)
                install_cron = bool(body.get("install_cron", False))
                with self.state.lock:
                    self.state.engine.schedule(cycles=cycles, install_cron=install_cron)
                if install_cron:
                    self._send_json({"status": "ok", "message": "Cron kurulumu tamamlandi"})
                else:
                    self._send_json({"status": "ok", "message": f"Zamanlayici dongusu tamamlandi ({cycles} tur)"})
                return

            self._send_json({"status": "error", "message": "Bulunamadi"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"status": "error", "message": f"Sunucu hatasi: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quiet default HTTP access logs; engine logger already writes structured logs.
        return

    def _handle_health(self) -> None:
        with self.state.lock:
            accounts = self.state.engine.account_manager.list_accounts(only_active=False)
        uptime_seconds = int(time.time() - self.state.started_at)
        payload = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "accounts_total": len(accounts),
            "accounts_active": len([a for a in accounts if a.active]),
        }
        self._send_json({"status": "ok", "data": payload})

    def _handle_accounts(self) -> None:
        with self.state.lock:
            accounts = self.state.engine.account_manager.list_accounts(only_active=False)
            raw_cfg = self._load_config()
            raw_accounts = raw_cfg.get("accounts", []) if isinstance(raw_cfg.get("accounts", []), list) else []
            raw_map = {
                str(item.get("name", "")).strip(): item
                for item in raw_accounts
                if isinstance(item, dict)
            }
        payload = [
            {
                "name": account.name,
                "username_or_email": str(raw_map.get(account.name, {}).get("username_or_email", account.name)),
                "niche": account.niche,
                "language": account.language,
                "timezone": account.timezone,
                "ollama_model": account.ollama_model,
                "active": account.active,
                "session_state_path": str(account.session_state_path),
            }
            for account in accounts
        ]
        self._send_json({"status": "ok", "data": payload})

    def _handle_account_upsert(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.state.lock:
            config = self._load_config()
            accounts = config.setdefault("accounts", [])
            if not isinstance(accounts, list):
                raise RuntimeError("Config section 'accounts' gecersiz")

            existing_idx = self._find_account_index(accounts, body)
            existing = accounts[existing_idx] if existing_idx is not None else None
            account = self._normalize_account_payload(body, existing=existing)

            if existing_idx is None:
                accounts.append(account)
            else:
                accounts[existing_idx] = account

            self._save_config(config)
            self._reload_engine()
        return account

    def _handle_account_delete(self, body: dict[str, Any]) -> str:
        login = str(body.get("username_or_email", body.get("name", ""))).strip()
        if not login:
            raise RuntimeError("Silinecek hesap icin kullanici adi veya e-posta zorunlu")

        with self.state.lock:
            config = self._load_config()
            accounts = config.setdefault("accounts", [])
            if not isinstance(accounts, list):
                raise RuntimeError("Config section 'accounts' gecersiz")

            target_slug = self._slugify_name(login)
            kept = []
            removed_name: str | None = None
            for item in accounts:
                item_name = str(item.get("name", "")).strip()
                item_login = str(item.get("username_or_email", item_name)).strip()
                same = (
                    item_name == login
                    or item_login == login
                    or item_name == target_slug
                )
                if same and removed_name is None:
                    removed_name = item_name
                    continue
                kept.append(item)

            if removed_name is None:
                raise RuntimeError(f"Hesap bulunamadi: {login}")
            accounts = kept
            if not accounts:
                raise RuntimeError("En az bir hesap kalmali")

            config["accounts"] = accounts
            self._save_config(config)
            self._reload_engine()

        return removed_name

    def _normalize_account_payload(self, body: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
        username_or_email = str(body.get("username_or_email", body.get("name", ""))).strip()
        if not username_or_email:
            raise RuntimeError("Kullanici adi veya e-posta zorunlu")

        password = str(body.get("password", "")).strip()
        if not password:
            raise RuntimeError("Sifre zorunlu")

        base = existing or {}
        name = str(base.get("name", "")).strip() or self._slugify_name(username_or_email)
        if not name:
            raise RuntimeError("Hesap kodu olusturulamadi")

        return {
            "name": name,
            "username_or_email": username_or_email,
            "password": password,
            "active": self._to_bool(base.get("active", True)),
            "niche": str(base.get("niche", "general")),
            "language": str(base.get("language", "en")),
            "timezone": str(base.get("timezone", "UTC")),
            "ollama_model": str(base.get("ollama_model", "llama3.1:8b")),
            "tags_seed": list(base.get("tags_seed", [])),
            "session_state_path": str(base.get("session_state_path", f"storage/accounts/{name}/session_state.json")),
        }

    def _find_account_index(self, accounts: list[dict[str, Any]], body: dict[str, Any]) -> int | None:
        login = str(body.get("username_or_email", body.get("name", ""))).strip()
        if not login:
            return None
        target_slug = self._slugify_name(login)
        for idx, item in enumerate(accounts):
            item_name = str(item.get("name", "")).strip()
            item_login = str(item.get("username_or_email", item_name)).strip()
            if item_name == login or item_login == login or item_name == target_slug:
                return idx
        return None

    def _slugify_name(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug[:64]

    def _to_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized in {"1", "true", "yes", "on", "evet"}
        return False

    def _handle_stats(self, account_name: str | None) -> None:
        with self.state.lock:
            payload = self.state.engine.stats(account_name=account_name)
        self._send_json({"status": "ok", "data": payload})

    def _handle_config(self) -> None:
        with self.state.lock:
            cfg = self.state.engine.config
        payload = {
            "execution": cfg.get("execution", {}),
            "scheduler": cfg.get("scheduler", {}),
            "upload": {
                "mode": cfg.get("upload", {}).get("mode"),
                "retry_attempts": cfg.get("upload", {}).get("retry_attempts"),
                "retry_backoff_seconds": cfg.get("upload", {}).get("retry_backoff_seconds"),
                "mock_mode": cfg.get("upload", {}).get("mock_mode"),
            },
            "analytics": cfg.get("analytics", {}),
            "strategy": {
                "learning_rate": cfg.get("strategy", {}).get("learning_rate"),
                "underperforming_threshold": cfg.get("strategy", {}).get("underperforming_threshold"),
                "rotation_floor": cfg.get("strategy", {}).get("rotation_floor"),
            },
        }
        self._send_json({"status": "ok", "data": payload})

    def _handle_logs(self, lines: int = 80) -> None:
        log_path = self.state.engine.storage_root / "logs" / "autonomous_engine.jsonl"
        if not log_path.exists():
            self._send_json({"status": "ok", "data": []})
            return

        tail = deque(maxlen=lines)
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    tail.append(stripped)

        parsed: list[dict[str, Any]] = []
        for line in tail:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                parsed.append({"message": line, "level": "UNKNOWN"})

        self._send_json({"status": "ok", "data": parsed})

    def _serve_static(self, relative_path: str) -> None:
        root = self.static_root.resolve()
        target = (root / relative_path).resolve()

        if not str(target).startswith(str(root)):
            self._send_json({"status": "error", "message": "Gecersiz yol"}, status=HTTPStatus.FORBIDDEN)
            return
        if not target.exists() or not target.is_file():
            self._send_json({"status": "error", "message": "Dosya bulunamadi"}, status=HTTPStatus.NOT_FOUND)
            return

        mime_type, _ = mimetypes.guess_type(str(target))
        if not mime_type:
            mime_type = "application/octet-stream"

        content = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if not raw_length:
            return {}
        try:
            length = int(raw_length)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def _safe_int(self, value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def _load_config(self) -> dict[str, Any]:
        if not self.state.config_path.exists():
            raise RuntimeError(f"Config dosyasi bulunamadi: {self.state.config_path}")
        payload = yaml.safe_load(self.state.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise RuntimeError("Config YAML icerigi gecersiz")
        return payload

    def _save_config(self, payload: dict[str, Any]) -> None:
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
        self.state.config_path.write_text(text, encoding="utf-8")

    def _reload_engine(self) -> None:
        from engine import AutonomousGrowthEngine

        self.state.engine = AutonomousGrowthEngine(config_path=str(self.state.config_path))


def run_dashboard_server(config_path: str, host: str = "127.0.0.1", port: int = 8787) -> None:
    """Start dashboard web server."""
    project_root = Path(__file__).resolve().parent.parent
    static_root = project_root / "web_dashboard" / "static"
    from engine import AutonomousGrowthEngine

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (project_root / config_file).resolve()

    engine = AutonomousGrowthEngine(config_path=str(config_file))
    state = DashboardState(engine=engine, config_path=config_file, started_at=time.time(), lock=threading.Lock())

    handler = type(
        "BoundDashboardHandler",
        (DashboardHandler,),
        {"state": state, "static_root": static_root},
    )
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        raise RuntimeError(
            f"Dashboard port bind failed on {host}:{port}. "
            f"Port may be in use or host may be invalid. Original error: {exc}"
        ) from exc

    print(f"[dashboard] Serving on http://{host}:{port}")
    print("[dashboard] Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[dashboard] Server stopped")


def main() -> None:
    """CLI entrypoint for standalone dashboard launch."""
    parser = argparse.ArgumentParser(description="Autonomous Growth Engine Dashboard")
    parser.add_argument("--config", default="config.yaml", help="Path to engine config YAML")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8787, help="Bind port")
    args = parser.parse_args()

    run_dashboard_server(config_path=args.config, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
