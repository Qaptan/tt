"""CLI entrypoint for the autonomous TikTok growth engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.table import Table
except ModuleNotFoundError:  # pragma: no cover - fallback for bare Python environments
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

if Console is not None:
    console = Console()
else:
    class _PlainConsole:
        def print(self, message: Any) -> None:
            print(message)

        def print_json(self, payload: str) -> None:
            print(payload)

    console = _PlainConsole()


def _maybe_reexec_in_venv() -> None:
    """Re-run CLI using local .venv interpreter when available."""
    if os.getenv("TIKTOK_SKIP_VENV_REEXEC") == "1":
        return

    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    try:
        current_prefix = Path(sys.prefix).resolve()
        target_prefix = (project_root / ".venv").resolve()
    except FileNotFoundError:
        return

    if current_prefix == target_prefix:
        return

    target_python = venv_python
    os.environ["TIKTOK_SKIP_VENV_REEXEC"] = "1"
    os.execv(str(target_python), [str(target_python), str(project_root / "main.py"), *sys.argv[1:]])


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Autonomous TikTok Growth Engine")
    parser.add_argument(
        "--config",
        default=os.getenv("TIKTOK_ENGINE_CONFIG", "config.yaml"),
        help="Path to config YAML",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize storage and account state")

    run_parser = sub.add_parser("run", help="Run end-to-end automation")
    run_parser.add_argument("--account", help="Run only one account")

    schedule_parser = sub.add_parser("schedule", help="Run scheduler loop or install cron")
    schedule_parser.add_argument("--cycles", type=int, default=1, help="Number of scheduler cycles")
    schedule_parser.add_argument("--install-cron", action="store_true", help="Install cron line")

    stats_parser = sub.add_parser("stats", help="Show account stats")
    stats_parser.add_argument("--account", help="Show one account")

    analyze_parser = sub.add_parser("analyze", help="Analyze metrics")
    analyze_parser.add_argument("--account", help="Analyze one account")

    retrain_parser = sub.add_parser("retrain", help="Retrain strategy model from metrics")
    retrain_parser.add_argument("--account", help="Retrain one account")

    web_parser = sub.add_parser("web", help="Start web dashboard")
    web_parser.add_argument("--host", default="127.0.0.1", help="Dashboard bind host")
    web_parser.add_argument("--port", type=int, default=8787, help="Dashboard bind port")

    return parser


def print_run_results(results: list[Any]) -> None:
    """Render run results in rich table."""
    if Table is None:
        for row in results:
            console.print(
                f"{row.account}: generated={row.generated} uploaded={row.uploaded} failed_uploads={row.failed_uploads} message={row.message}"
            )
        return

    table = Table(title="Run Results")
    table.add_column("Account")
    table.add_column("Generated", justify="right")
    table.add_column("Uploaded", justify="right")
    table.add_column("Failed Uploads", justify="right")
    table.add_column("Message")

    for row in results:
        table.add_row(
            row.account,
            str(row.generated),
            str(row.uploaded),
            str(row.failed_uploads),
            row.message,
        )
    console.print(table)


def print_stats(stats_rows: list[dict[str, Any]]) -> None:
    """Render stats rows."""
    if Table is None:
        console.print(json.dumps(stats_rows, indent=2))
        return

    table = Table(title="Account Stats")
    table.add_column("Account")
    table.add_column("Generated", justify="right")
    table.add_column("Uploaded", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Queue Pending", justify="right")
    table.add_column("Metrics Pending", justify="right")

    for row in stats_rows:
        table.add_row(
            row["account"],
            str(row["generated"]),
            str(row["uploaded"]),
            str(row["failed_uploads"]),
            str(row["queue_pending"]),
            str(row["metrics_pending"]),
        )
    console.print(table)


def _load_engine_class() -> Any:
    """Import engine lazily so CLI can show a clear setup message when dependencies are missing."""
    try:
        from engine import AutonomousGrowthEngine
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        console.print(f"Eksik Python paketi: {missing}")
        console.print("Sanal ortam ile çalıştırın:")
        console.print("  cd tiktok-autonomous-growth-engine")
        console.print("  source .venv/bin/activate")
        console.print("  pip install -r requirements.txt")
        console.print("  python main.py web --host 0.0.0.0 --port 8787")
        raise SystemExit(1) from exc
    return AutonomousGrowthEngine


def main() -> None:
    """Program entrypoint."""
    _maybe_reexec_in_venv()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        EngineClass = _load_engine_class()
        engine = EngineClass(config_path=args.config)
        engine.init()
        console.print("Initialization complete")
        return

    if args.command == "run":
        EngineClass = _load_engine_class()
        engine = EngineClass(config_path=args.config)
        results = engine.run(account_name=args.account)
        print_run_results(results)
        return

    if args.command == "schedule":
        EngineClass = _load_engine_class()
        engine = EngineClass(config_path=args.config)
        engine.schedule(cycles=args.cycles, install_cron=args.install_cron)
        console.print("Scheduler execution complete")
        return

    if args.command == "stats":
        EngineClass = _load_engine_class()
        engine = EngineClass(config_path=args.config)
        rows = engine.stats(account_name=args.account)
        print_stats(rows)
        return

    if args.command == "analyze":
        EngineClass = _load_engine_class()
        engine = EngineClass(config_path=args.config)
        analysis = engine.analyze(account_name=args.account)
        console.print_json(json.dumps(analysis, indent=2))
        return

    if args.command == "retrain":
        EngineClass = _load_engine_class()
        engine = EngineClass(config_path=args.config)
        update = engine.retrain(account_name=args.account)
        console.print_json(json.dumps(update, indent=2))
        return

    if args.command == "web":
        try:
            from web_dashboard.server import run_dashboard_server
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            console.print(f"Eksik Python paketi: {missing}")
            console.print("Sanal ortam ile çalıştırın:")
            console.print("  cd tiktok-autonomous-growth-engine")
            console.print("  source .venv/bin/activate")
            console.print("  pip install -r requirements.txt")
            console.print("  python main.py web --host 0.0.0.0 --port 8787")
            raise SystemExit(1) from exc

        try:
            run_dashboard_server(config_path=args.config, host=args.host, port=args.port)
        except ModuleNotFoundError as exc:
            missing = exc.name or "unknown"
            console.print(f"Eksik Python paketi: {missing}")
            console.print("Sanal ortam ile çalıştırın:")
            console.print("  cd tiktok-autonomous-growth-engine")
            console.print("  source .venv/bin/activate")
            console.print("  pip install -r requirements.txt")
            console.print("  python main.py web --host 0.0.0.0 --port 8787")
            raise SystemExit(1) from exc
        return


if __name__ == "__main__":
    main()
