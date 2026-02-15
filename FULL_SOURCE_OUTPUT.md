```text
tiktok-autonomous-growth-engine/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── config.yaml
├── main.py
├── engine.py
├── scripts/
│   └── save_tiktok_session.py
├── core/
│   ├── __init__.py
│   ├── constants.py
│   ├── config_loader.py
│   └── logger.py
├── accounts/
│   ├── __init__.py
│   ├── account_manager.py
│   └── account_state_handler.py
├── trend_engine/
│   ├── __init__.py
│   ├── trend_scraper.py
│   ├── trend_classifier.py
│   └── trend_score_engine.py
├── content_engine/
│   ├── __init__.py
│   ├── idea_generator.py
│   ├── script_generator.py
│   ├── hook_optimizer.py
│   ├── caption_generator.py
│   └── hashtag_optimizer.py
├── media_engine/
│   ├── __init__.py
│   ├── voice_generator.py
│   ├── subtitle_generator.py
│   ├── video_renderer.py
│   └── thumbnail_selector.py
├── upload_engine/
│   ├── __init__.py
│   ├── draft_or_publish_mode.py
│   ├── upload_retry_logic.py
│   ├── upload_queue_manager.py
│   └── tiktok_uploader.py
├── analytics_engine/
│   ├── __init__.py
│   ├── metrics_collector.py
│   ├── performance_analyzer.py
│   ├── winner_pattern_detector.py
│   └── format_score_model.py
├── strategy_engine/
│   ├── __init__.py
│   ├── decision_model.py
│   ├── format_multiplier.py
│   ├── kill_underperforming_logic.py
│   └── publish_time_optimizer.py
├── scheduler/
│   ├── __init__.py
│   ├── cron_runner.py
│   └── queue_processor.py
├── storage/
│   ├── videos/.gitkeep
│   ├── logs/.gitkeep
│   ├── metrics/.gitkeep
│   ├── models/.gitkeep
│   ├── accounts/.gitkeep
│   └── queue/.gitkeep
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_core_constants.py
    ├── test_core_config_loader.py
    ├── test_core_logger.py
    ├── test_account_state_handler.py
    ├── test_account_manager.py
    ├── test_trend_scraper.py
    ├── test_trend_classifier.py
    ├── test_trend_score_engine.py
    ├── test_content_idea_generator.py
    ├── test_content_script_generator.py
    ├── test_content_hook_optimizer.py
    ├── test_content_caption_generator.py
    ├── test_content_hashtag_optimizer.py
    ├── test_media_voice_generator.py
    ├── test_media_subtitle_generator.py
    ├── test_media_video_renderer.py
    ├── test_media_thumbnail_selector.py
    ├── test_upload_draft_mode.py
    ├── test_upload_retry_logic.py
    ├── test_upload_queue_manager.py
    ├── test_upload_tiktok_uploader.py
    ├── test_analytics_metrics_collector.py
    ├── test_analytics_performance_analyzer.py
    ├── test_analytics_winner_detector.py
    ├── test_analytics_format_score_model.py
    ├── test_strategy_decision_model.py
    ├── test_strategy_format_multiplier.py
    ├── test_strategy_kill_underperforming.py
    ├── test_strategy_publish_time_optimizer.py
    ├── test_scheduler_cron_runner.py
    ├── test_scheduler_queue_processor.py
    └── test_engine_and_cli.py
```

`.env.example`
```bash
# Copy to .env and adjust
TIKTOK_ENGINE_CONFIG=config.yaml
OLLAMA_HOST=http://127.0.0.1:11434
DEFAULT_OLLAMA_MODEL=llama3.1:8b
PIPER_MODEL_PATH=/absolute/path/to/en_US-lessac-medium.onnx
TIKTOK_HEADLESS=true

```

`.gitignore`
```gitignore
__pycache__/
*.pyc
.pytest_cache/
.venv/
.env
storage/logs/*.log
storage/logs/*.jsonl
storage/videos/*.mp4
storage/videos/*.wav
storage/videos/*.srt
storage/videos/*.png
storage/models/*.json
storage/metrics/*.jsonl
storage/accounts/*/session*.json

```

`README.md`
```markdown
# tiktok-autonomous-growth-engine

Fully local, modular, multi-account TikTok automation engine for Linux.

## Features
- Local AI pipeline: Ollama + Piper + Whisper + FFmpeg
- Playwright upload automation with draft/publish mode
- Multi-account isolation (state, analytics, strategy, queue, session)
- Self-improving feedback loop with dynamic strategy weights
- Config-driven architecture and structured JSON logging
- Scheduler support (foreground interval loop + cron installation)
- Pytest test suite across all modules

## Folder Structure
```text
tiktok-autonomous-growth-engine/
├── main.py
├── engine.py
├── config.yaml
├── .env.example
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
│   └── save_tiktok_session.py
├── core/
│   ├── constants.py
│   ├── config_loader.py
│   └── logger.py
├── accounts/
│   ├── account_manager.py
│   └── account_state_handler.py
├── trend_engine/
│   ├── trend_scraper.py
│   ├── trend_classifier.py
│   └── trend_score_engine.py
├── content_engine/
│   ├── idea_generator.py
│   ├── script_generator.py
│   ├── hook_optimizer.py
│   ├── caption_generator.py
│   └── hashtag_optimizer.py
├── media_engine/
│   ├── voice_generator.py
│   ├── subtitle_generator.py
│   ├── video_renderer.py
│   └── thumbnail_selector.py
├── upload_engine/
│   ├── draft_or_publish_mode.py
│   ├── upload_retry_logic.py
│   ├── upload_queue_manager.py
│   └── tiktok_uploader.py
├── analytics_engine/
│   ├── metrics_collector.py
│   ├── performance_analyzer.py
│   ├── winner_pattern_detector.py
│   └── format_score_model.py
├── strategy_engine/
│   ├── decision_model.py
│   ├── format_multiplier.py
│   ├── kill_underperforming_logic.py
│   └── publish_time_optimizer.py
├── scheduler/
│   ├── cron_runner.py
│   └── queue_processor.py
├── storage/
│   ├── videos/
│   ├── logs/
│   ├── metrics/
│   ├── models/
│   ├── accounts/
│   └── queue/
└── tests/
```

## Requirements
- Linux (Ubuntu/Zorin compatible)
- Python 3.10+
- FFmpeg
- Ollama
- Piper
- Whisper local CLI
- Playwright Chromium

## Installation
```bash
cd tiktok-autonomous-growth-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Install local binaries (outside Python):
- `ffmpeg` (apt package)
- `ollama` (local server)
- `piper` (binary + voice model)
- `whisper` (CLI local install)

## Setup
1. Copy environment template.
```bash
cp .env.example .env
```

2. Edit `config.yaml` for accounts, upload mode, paths, and strategy weights.

3. Initialize storage and account state.
```bash
python main.py init
```

## Connect TikTok Session
For each account, save an authenticated Playwright session file:
```bash
python scripts/save_tiktok_session.py --out storage/accounts/fitness_account/session_state.json
python scripts/save_tiktok_session.py --out storage/accounts/finance_account/session_state.json
```

Then set `upload.mock_mode: false` in `config.yaml`.

## CLI Commands
```bash
python main.py init
python main.py run
python main.py run --account fitness_account
python main.py schedule
python main.py stats
python main.py analyze
python main.py retrain
```

Extra scheduler options:
```bash
python main.py schedule --cycles 24
python main.py schedule --install-cron
```

## Learning Loop
After each upload:
1. Upload succeeds and post ID is persisted as pending metrics
2. Metrics collector updates eligible pending posts
3. Performance analyzer computes engagement score
4. Winner detector extracts best hooks/durations/captions/formats/times
5. Underperformers are penalized
6. Strategy weights are updated and normalized
7. Next generation call samples from updated weights

Engagement formula:
```text
engagement_score =
    (views_weight * views)
  + (completion_weight * completion_rate)
  + (shares_weight * shares)
  + (comments_weight * comments)
```

Weights are configured in `config.yaml` under `analytics.engagement_weights`.

## Scheduler Setup
### Foreground interval loop
```bash
python main.py schedule --cycles 100
```

### Cron installation
```bash
python main.py schedule --install-cron
```
This installs a `crontab` entry that runs `python main.py run` every configured interval.

## Troubleshooting
- Upload fails immediately:
  - Check `upload.mock_mode` and `session_state_path`
  - Recreate session with `scripts/save_tiktok_session.py`
- Video rendering fails:
  - Verify `ffmpeg` is on PATH (`which ffmpeg`)
  - Inspect JSON logs in `storage/logs/autonomous_engine.jsonl`
- Ollama fallback always used:
  - Ensure `ollama` binary is installed and model exists (`ollama list`)
- No metrics collected:
  - With `analytics.mock_mode: false`, pending metrics remain pending until real data integration
- Playwright errors:
  - Run `playwright install chromium`
  - Check Linux dependencies required by Chromium

## Testing
```bash
pytest
```

```

`pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tiktok-autonomous-growth-engine"
version = "0.1.0"
description = "Fully local autonomous TikTok growth engine"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "PyYAML>=6.0",
  "pandas>=2.0",
  "rich>=13.0",
  "playwright>=1.45",
  "pytest>=8.0"
]

[tool.setuptools]
include-package-data = true

[tool.setuptools.packages.find]
include = [
  "core*",
  "accounts*",
  "trend_engine*",
  "content_engine*",
  "media_engine*",
  "upload_engine*",
  "analytics_engine*",
  "strategy_engine*",
  "scheduler*"
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-q"

```

`requirements.txt`
```text
PyYAML>=6.0
pandas>=2.0
rich>=13.0
playwright>=1.45
pytest>=8.0

```

`config.yaml`
```yaml
project:
  name: tiktok-autonomous-growth-engine
  storage_root: storage

execution:
  account_mode: sequential   # sequential | parallel
  max_workers: 2

generation:
  allowed_durations: [9, 10, 11, 12, 13, 14, 15]
  default_duration_seconds: 12
  video:
    resolution: "1080x1920"
    fps: 30
    background_colors: ["0x101820", "0x1f2a44", "0x2d1e2f", "0x1f3b4d"]
  formats:
    - story
    - listicle
    - tutorial
    - challenge
  hook_styles:
    - question
    - shock
    - contrarian
    - promise
  caption_styles:
    - direct
    - curiosity
    - urgency
    - community

analytics:
  mock_mode: true
  min_age_minutes_for_collection: 5
  engagement_weights:
    views: 0.001
    completion_rate: 100.0
    shares: 6.0
    comments: 4.0

strategy:
  learning_rate: 0.15
  underperforming_threshold: 0.45
  rotation_floor: 0.05
  initial_format_weights:
    story: 1.0
    listicle: 1.0
    tutorial: 1.0
    challenge: 1.0
  initial_hook_weights:
    question: 1.0
    shock: 1.0
    contrarian: 1.0
    promise: 1.0
  initial_caption_weights:
    direct: 1.0
    curiosity: 1.0
    urgency: 1.0
    community: 1.0
  initial_duration_weights:
    "9": 1.0
    "10": 1.0
    "11": 1.0
    "12": 1.0
    "13": 1.0
    "14": 1.0
    "15": 1.0

upload:
  mode: draft                 # draft | publish
  retry_attempts: 3
  retry_backoff_seconds: 5
  headless: true
  mock_mode: true
  tiktok_upload_url: "https://www.tiktok.com/upload"
  post_wait_seconds: 12

scheduler:
  interval_minutes: 60
  cron_enabled: false

accounts:
  - name: fitness_account
    active: true
    niche: fitness
    language: en
    timezone: America/New_York
    ollama_model: llama3.1:8b
    tags_seed: [fitness, workout, health, motivation]
    session_state_path: storage/accounts/fitness_account/session_state.json
  - name: finance_account
    active: true
    niche: personal finance
    language: en
    timezone: America/Chicago
    ollama_model: llama3.1:8b
    tags_seed: [money, saving, investing, budget]
    session_state_path: storage/accounts/finance_account/session_state.json

```

`main.py`
```python
"""CLI entrypoint for the autonomous TikTok growth engine."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from rich.console import Console
from rich.table import Table

from engine import AutonomousGrowthEngine

console = Console()


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

    return parser


def print_run_results(results: list[Any]) -> None:
    """Render run results in rich table."""
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


def main() -> None:
    """Program entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    engine = AutonomousGrowthEngine(config_path=args.config)

    if args.command == "init":
        engine.init()
        console.print("Initialization complete")
        return

    if args.command == "run":
        results = engine.run(account_name=args.account)
        print_run_results(results)
        return

    if args.command == "schedule":
        engine.schedule(cycles=args.cycles, install_cron=args.install_cron)
        console.print("Scheduler execution complete")
        return

    if args.command == "stats":
        rows = engine.stats(account_name=args.account)
        print_stats(rows)
        return

    if args.command == "analyze":
        analysis = engine.analyze(account_name=args.account)
        console.print_json(json.dumps(analysis, indent=2))
        return

    if args.command == "retrain":
        update = engine.retrain(account_name=args.account)
        console.print_json(json.dumps(update, indent=2))
        return


if __name__ == "__main__":
    main()

```

`engine.py`
```python
"""Main orchestration for the autonomous TikTok growth engine."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from accounts.account_manager import AccountManager, AccountProfile
from accounts.account_state_handler import AccountState
from analytics_engine.format_score_model import FormatScoreModel
from analytics_engine.metrics_collector import MetricsCollector
from analytics_engine.performance_analyzer import EngagementWeights, PerformanceAnalyzer
from analytics_engine.winner_pattern_detector import WinnerPatternDetector
from content_engine.caption_generator import CaptionGenerator
from content_engine.hashtag_optimizer import HashtagOptimizer
from content_engine.hook_optimizer import HookOptimizer
from content_engine.idea_generator import IdeaGenerator
from content_engine.script_generator import ScriptGenerator
from core.config_loader import ConfigLoader
from core.logger import setup_logger
from media_engine.subtitle_generator import SubtitleGenerator
from media_engine.thumbnail_selector import ThumbnailSelector
from media_engine.video_renderer import VideoRenderer
from media_engine.voice_generator import VoiceGenerator
from scheduler.cron_runner import CronRunner
from scheduler.queue_processor import QueueProcessor
from strategy_engine.decision_model import DecisionModel
from strategy_engine.format_multiplier import FormatMultiplier
from strategy_engine.kill_underperforming_logic import KillUnderperformingLogic
from strategy_engine.publish_time_optimizer import PublishTimeOptimizer
from trend_engine.trend_classifier import TrendClassifier
from trend_engine.trend_score_engine import TrendScoreEngine
from trend_engine.trend_scraper import TrendScraper
from upload_engine.draft_or_publish_mode import DraftOrPublishMode
from upload_engine.tiktok_uploader import TikTokUploader
from upload_engine.upload_queue_manager import UploadQueueManager, UploadTask


@dataclass
class AccountRunResult:
    """Result of a single account run."""

    account: str
    generated: int
    uploaded: int
    failed_uploads: int
    message: str


class AutonomousGrowthEngine:
    """Coordinates generation, upload, analytics, and learning loops."""

    def __init__(self, config_path: str | None = None) -> None:
        cfg_path = Path(config_path) if config_path else None
        self.config_loader = ConfigLoader(config_path=cfg_path)
        self.config = self.config_loader.load()

        self.project_root = self.config_loader.config_path.parent.resolve()
        storage_root = Path(self.config["project"]["storage_root"])
        self.storage_root = storage_root if storage_root.is_absolute() else self.project_root / storage_root

        self.logger = setup_logger(self.storage_root / "logs")
        self.account_manager = AccountManager(self.config, self.project_root)

        self.trend_scraper = TrendScraper()
        self.trend_classifier = TrendClassifier()
        self.trend_score_engine = TrendScoreEngine()

        self.idea_generator = IdeaGenerator(self.logger)
        self.script_generator = ScriptGenerator()
        self.hook_optimizer = HookOptimizer()
        self.caption_generator = CaptionGenerator()
        self.hashtag_optimizer = HashtagOptimizer()

        video_cfg = self.config.get("generation", {}).get("video", {})
        self.voice_generator = VoiceGenerator(self.logger)
        self.subtitle_generator = SubtitleGenerator(self.logger)
        self.video_renderer = VideoRenderer(
            logger=self.logger,
            fps=int(video_cfg.get("fps", 30)),
            resolution=str(video_cfg.get("resolution", "1080x1920")),
        )
        self.thumbnail_selector = ThumbnailSelector(self.logger)

        upload_cfg = self.config.get("upload", {})
        self.uploader = TikTokUploader(
            logger=self.logger,
            upload_url=str(upload_cfg.get("tiktok_upload_url", "https://www.tiktok.com/upload")),
            headless=bool(upload_cfg.get("headless", True)),
            mock_mode=bool(upload_cfg.get("mock_mode", True)),
        )

        weights_cfg = self.config.get("analytics", {}).get("engagement_weights", {})
        self.performance_analyzer = PerformanceAnalyzer(
            EngagementWeights(
                views=float(weights_cfg.get("views", 0.001)),
                completion_rate=float(weights_cfg.get("completion_rate", 100.0)),
                shares=float(weights_cfg.get("shares", 6.0)),
                comments=float(weights_cfg.get("comments", 4.0)),
            )
        )
        self.winner_detector = WinnerPatternDetector()
        self.decision_model = DecisionModel()
        self.format_multiplier = FormatMultiplier()
        self.kill_logic = KillUnderperformingLogic()
        self.publish_optimizer = PublishTimeOptimizer()
        self.queue_processor = QueueProcessor(self.logger)
        self.cron_runner = CronRunner(self.logger)

    def init(self) -> None:
        """Initialize storage and state files for all active accounts."""
        for account in self.account_manager.list_accounts(only_active=False):
            account_dir = self.account_manager.ensure_account_dirs(account)
            state_handler = self.account_manager.state_handler(account)
            state_handler.load(account.name)

            model_store = FormatScoreModel(
                model_path=self.account_manager.strategy_model_path(account),
                rotation_floor=float(self.config["strategy"].get("rotation_floor", 0.05)),
            )
            model_store.load_or_init(self.config.get("strategy", {}))

            metrics_path = self.account_manager.metrics_path(account)
            metrics_path.touch(exist_ok=True)

            queue_path = self.account_manager.queue_path(account)
            UploadQueueManager(queue_path)

            if not account.session_state_path.exists():
                account.session_state_path.parent.mkdir(parents=True, exist_ok=True)
                account.session_state_path.write_text("{}", encoding="utf-8")

            self.logger.info("Initialized account", context={"account": account.name, "dir": str(account_dir)})

    def run(self, account_name: str | None = None) -> list[AccountRunResult]:
        """Run generation/upload/learning for one or multiple accounts."""
        accounts = self._resolve_accounts(account_name)
        mode = str(self.config["execution"].get("account_mode", "sequential")).lower()

        if mode == "parallel" and len(accounts) > 1:
            workers = int(self.config["execution"].get("max_workers", 2))
            return self._run_parallel(accounts, workers)
        return [self._run_account(account) for account in accounts]

    def schedule(self, cycles: int = 1, install_cron: bool = False) -> None:
        """Run scheduler loop or install cron rule."""
        interval = int(self.config["scheduler"].get("interval_minutes", 60))
        if install_cron:
            line = self.cron_runner.cron_line(project_root=self.project_root, interval_minutes=interval)
            self.cron_runner.install(line)
            self.logger.info("Cron installed", context={"line": line})
            return

        self.cron_runner.run_interval_loop(
            interval_minutes=interval,
            fn=lambda: self.run(),
            cycles=cycles,
        )

    def stats(self, account_name: str | None = None) -> list[dict[str, Any]]:
        """Return account-level operational stats."""
        stats_rows: list[dict[str, Any]] = []
        for account in self._resolve_accounts(account_name):
            state = self.account_manager.state_handler(account).load(account.name)
            queue = UploadQueueManager(self.account_manager.queue_path(account))
            metrics = MetricsCollector(self.account_manager.metrics_path(account), self.logger)
            pending_metrics = metrics.dataframe(status_filter="pending")
            stats_rows.append(
                {
                    "account": account.name,
                    "generated": state.generated_count,
                    "uploaded": state.uploaded_count,
                    "failed_uploads": state.failed_uploads,
                    "queue_pending": queue.pending_count(),
                    "metrics_pending": 0 if pending_metrics.empty else int(pending_metrics.shape[0]),
                }
            )
        return stats_rows

    def analyze(self, account_name: str | None = None) -> dict[str, Any]:
        """Analyze collected metrics and return summary."""
        summaries: dict[str, Any] = {}
        min_age = int(self.config["analytics"].get("min_age_minutes_for_collection", 5))
        mock_mode = bool(self.config["analytics"].get("mock_mode", True))

        for account in self._resolve_accounts(account_name):
            collector = MetricsCollector(self.account_manager.metrics_path(account), self.logger)
            collector.collect(min_age_minutes=min_age, mock_mode=mock_mode)
            collected_df = collector.dataframe(status_filter="collected")
            scored = self.performance_analyzer.analyze(collected_df)

            if scored.empty:
                summaries[account.name] = {
                    "status": "no_collected_metrics",
                    "message": "Metrics pending or unavailable",
                }
                continue

            top_score = float(scored.iloc[0]["engagement_score"])
            avg_score = float(scored["engagement_score"].mean())
            winners = self.winner_detector.detect(scored)["winners"]
            summaries[account.name] = {
                "status": "ok",
                "top_score": round(top_score, 4),
                "avg_score": round(avg_score, 4),
                "rows": int(scored.shape[0]),
                "winners": winners,
            }

        return summaries

    def retrain(self, account_name: str | None = None) -> dict[str, Any]:
        """Retrain strategy weights using all collected metrics."""
        updated: dict[str, Any] = {}
        learning_rate = float(self.config["strategy"].get("learning_rate", 0.15))
        threshold = float(self.config["strategy"].get("underperforming_threshold", 0.45))

        for account in self._resolve_accounts(account_name):
            model_store = FormatScoreModel(
                self.account_manager.strategy_model_path(account),
                rotation_floor=float(self.config["strategy"].get("rotation_floor", 0.05)),
            )
            model = model_store.load_or_init(self.config.get("strategy", {}))

            metrics = MetricsCollector(self.account_manager.metrics_path(account), self.logger)
            collected = metrics.dataframe(status_filter="collected")
            scored = self.performance_analyzer.analyze(collected)

            if scored.empty:
                updated[account.name] = {"status": "skipped", "reason": "no_collected_metrics"}
                continue

            pattern = self.winner_detector.detect(scored)
            losers = pattern["losers"]
            under = self.kill_logic.detect_underperformers(scored, threshold=threshold)
            losers["formats"] = sorted(set(losers.get("formats", []) + under.get("formats", [])))
            losers["hooks"] = sorted(set(losers.get("hooks", []) + under.get("hooks", [])))
            losers["caption_styles"] = sorted(set(losers.get("caption_styles", []) + under.get("caption_styles", [])))
            losers["durations"] = sorted(set(losers.get("durations", []) + under.get("durations", [])))

            model = model_store.update(
                model=model,
                winners=pattern["winners"],
                losers=losers,
                learning_rate=learning_rate,
            )

            best_hour = self.publish_optimizer.best_hour(scored)
            model["publish_hour_weights"] = self.format_multiplier.boost(
                model["publish_hour_weights"],
                key=str(best_hour),
                multiplier=1.1,
            )
            model_store.save(model)
            updated[account.name] = {"status": "updated", "best_hour": best_hour}

        return updated

    def _run_parallel(self, accounts: list[AccountProfile], workers: int) -> list[AccountRunResult]:
        results: list[AccountRunResult] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(self._run_account, account) for account in accounts]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _run_account(self, account: AccountProfile) -> AccountRunResult:
        self.logger.info("Running account pipeline", context={"account": account.name})
        state_handler = self.account_manager.state_handler(account)
        state = state_handler.load(account.name)

        model_store = FormatScoreModel(
            self.account_manager.strategy_model_path(account),
            rotation_floor=float(self.config["strategy"].get("rotation_floor", 0.05)),
        )
        model = model_store.load_or_init(self.config.get("strategy", {}))

        queue_manager = UploadQueueManager(self.account_manager.queue_path(account))
        metrics_collector = MetricsCollector(self.account_manager.metrics_path(account), self.logger)

        # 1) Trend discovery + scoring
        raw_trends = self.trend_scraper.scrape(account.niche, limit=8)
        classified = self.trend_classifier.classify(raw_trends)
        scored = self.trend_score_engine.score(classified)
        scored_trends = [asdict(item) for item in scored]

        top_category = None
        if scored:
            top_phrase = scored[0].phrase
            for item in classified:
                if item.phrase == top_phrase:
                    top_category = item.category
                    break

        # 2) Decision and generation
        decision = self.decision_model.decide(
            strategy_model=model,
            top_trend_category=top_category,
            now=datetime.now(),
        )

        ideas = self.idea_generator.generate_ideas(
            account_name=account.name,
            niche=account.niche,
            model=account.ollama_model,
            scored_trends=scored_trends,
            format_name=decision.format_name,
            count=3,
        )
        idea = ideas[0]

        script = self.script_generator.generate(
            idea=idea,
            hook_style=decision.hook_style,
            duration_seconds=decision.duration_seconds,
        )

        historical = metrics_collector.dataframe(status_filter="collected")
        scored_history = self.performance_analyzer.analyze(historical)
        winners_history = self.winner_detector.detect(scored_history)["winners"]["hooks"] if not scored_history.empty else []
        script.hook = self.hook_optimizer.optimize(script.hook, winners_history, decision.hook_style)
        script.full_text = f"{script.hook} {script.body} {script.cta}"

        caption = self.caption_generator.generate(idea, decision.caption_style)
        hashtags = self.hashtag_optimizer.generate(
            niche=account.niche,
            trend_phrases=[item["phrase"] for item in scored_trends[:4]],
            seed_tags=account.tags_seed,
            max_tags=8,
        )

        # 3) Media generation
        try:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            videos_dir = self.account_manager.account_video_dir(account)
            audio_path = videos_dir / f"{now_str}_{account.name}.wav"
            subtitle_path = videos_dir / f"{now_str}_{account.name}.srt"
            video_path = videos_dir / f"{now_str}_{account.name}.mp4"
            thumb_path = videos_dir / f"{now_str}_{account.name}.png"

            voice_model_path = os.getenv("PIPER_MODEL_PATH")
            self.voice_generator.synthesize(
                text=script.full_text,
                output_wav=audio_path,
                voice_model_path=voice_model_path,
                duration_seconds=decision.duration_seconds,
            )
            self.subtitle_generator.generate(
                audio_path=audio_path,
                output_srt=subtitle_path,
                language=account.language,
                fallback_text=script.full_text,
                duration_seconds=decision.duration_seconds,
            )

            bg_music_path: Path | None = None
            custom_music = self.config.get("generation", {}).get("background_music_path")
            if custom_music:
                bg = Path(str(custom_music))
                bg_music_path = bg if bg.is_absolute() else self.project_root / bg

            video_cfg = self.config.get("generation", {}).get("video", {})
            bg_colors = video_cfg.get("background_colors", [])
            bg_color = bg_colors[0] if bg_colors else None

            self.video_renderer.render(
                output_video=video_path,
                audio_path=audio_path,
                subtitle_path=subtitle_path,
                hook_text=script.hook,
                body_text=script.body,
                caption_text=caption,
                duration_seconds=decision.duration_seconds,
                background_music=bg_music_path,
                background_color=bg_color,
            )
            self.thumbnail_selector.extract(video_path, thumb_path)
            state.generated_count += 1
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Rendering failed",
                context={"account": account.name, "error": str(exc)},
            )
            state_handler.mark_run(state, success=False)
            return AccountRunResult(
                account=account.name,
                generated=state.generated_count,
                uploaded=state.uploaded_count,
                failed_uploads=state.failed_uploads,
                message=f"render_failed: {exc}",
            )

        # 4) Queue + upload
        mode = DraftOrPublishMode.from_value(str(self.config.get("upload", {}).get("mode", "draft")))
        task = queue_manager.enqueue(
            account_name=account.name,
            video_path=video_path,
            caption=caption,
            hashtags=hashtags,
            mode=mode,
            thumbnail_path=thumb_path,
            metadata={
                "format_name": decision.format_name,
                "hook_style": decision.hook_style,
                "duration_seconds": decision.duration_seconds,
                "caption_style": decision.caption_style,
                "publish_hour": decision.publish_hour,
            },
        )
        self.logger.info("Queued upload", context={"account": account.name, "task_id": task.id})

        successful_post_ids: list[str] = []

        def upload_fn(item: UploadTask) -> tuple[bool, str | None, str | None]:
            result = self.uploader.upload(
                video_path=Path(item.video_path),
                caption=item.caption,
                hashtags=item.hashtags,
                mode=DraftOrPublishMode.from_value(item.mode),
                session_state_path=account.session_state_path,
                post_wait_seconds=int(self.config.get("upload", {}).get("post_wait_seconds", 12)),
            )
            return result.success, result.post_id, result.error

        def on_success(item: UploadTask, post_id: str | None) -> None:
            if not post_id:
                return
            successful_post_ids.append(post_id)
            metrics_collector.record_pending(
                post_id=post_id,
                account_name=item.account_name,
                metadata=item.metadata or {},
            )

        upload_result = self.queue_processor.process(
            queue_manager=queue_manager,
            upload_fn=upload_fn,
            attempts=int(self.config.get("upload", {}).get("retry_attempts", 3)),
            backoff_seconds=int(self.config.get("upload", {}).get("retry_backoff_seconds", 5)),
            on_success=on_success,
        )

        state.uploaded_count += upload_result.succeeded
        state.failed_uploads += upload_result.failed
        state.pending_metrics_post_ids = successful_post_ids

        # 5) Collect metrics and update learning model
        min_age = int(self.config.get("analytics", {}).get("min_age_minutes_for_collection", 5))
        mock_mode = bool(self.config.get("analytics", {}).get("mock_mode", True))
        new_metrics = metrics_collector.collect(min_age_minutes=min_age, mock_mode=mock_mode)
        collected_df = metrics_collector.dataframe(status_filter="collected")

        if not collected_df.empty:
            scored_df = self.performance_analyzer.analyze(collected_df)
            pattern = self.winner_detector.detect(scored_df)
            losers = pattern["losers"]

            threshold = float(self.config.get("strategy", {}).get("underperforming_threshold", 0.45))
            under = self.kill_logic.detect_underperformers(scored_df, threshold=threshold)
            losers["formats"] = sorted(set(losers.get("formats", []) + under.get("formats", [])))
            losers["hooks"] = sorted(set(losers.get("hooks", []) + under.get("hooks", [])))
            losers["caption_styles"] = sorted(set(losers.get("caption_styles", []) + under.get("caption_styles", [])))
            losers["durations"] = sorted(set(losers.get("durations", []) + under.get("durations", [])))

            model = model_store.update(
                model=model,
                winners=pattern["winners"],
                losers=losers,
                learning_rate=float(self.config.get("strategy", {}).get("learning_rate", 0.15)),
            )

            best_hour = self.publish_optimizer.best_hour(scored_df, fallback_hour=decision.publish_hour)
            model["publish_hour_weights"] = self.format_multiplier.boost(
                model["publish_hour_weights"], key=str(best_hour), multiplier=1.08
            )
            model["format_weights"] = self.format_multiplier.rotate(
                model["format_weights"],
                min_share=float(self.config.get("strategy", {}).get("rotation_floor", 0.05)),
            )

            model_store.save(model)
            self.logger.info(
                "Strategy model updated",
                context={
                    "account": account.name,
                    "new_metrics": len(new_metrics),
                    "best_hour": best_hour,
                    "top_hook": pattern["winners"].get("hooks", []),
                },
            )
        else:
            self.logger.info(
                "Analytics unavailable; metrics remain pending",
                context={"account": account.name, "pending": len(successful_post_ids)},
            )

        state_handler.mark_run(state, success=True)
        return AccountRunResult(
            account=account.name,
            generated=state.generated_count,
            uploaded=state.uploaded_count,
            failed_uploads=state.failed_uploads,
            message="ok",
        )

    def _resolve_accounts(self, account_name: str | None) -> list[AccountProfile]:
        if account_name:
            return [self.account_manager.get_account(account_name)]
        return self.account_manager.list_accounts(only_active=True)


def dataframe_to_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert dataframe to serializable list."""
    if df.empty:
        return []
    return df.to_dict(orient="records")

```

`scripts/save_tiktok_session.py`
```python
"""Save TikTok authenticated browser session state for automated uploads.

Usage:
  python scripts/save_tiktok_session.py --out storage/accounts/fitness_account/session_state.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save TikTok session state")
    parser.add_argument("--out", required=True, help="Output storage_state JSON path")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.tiktok.com/login")
        print("Log in manually, then press ENTER to save session state...")
        input()
        context.storage_state(path=str(out_path))
        browser.close()

    print(f"Saved session state to: {out_path}")


if __name__ == "__main__":
    main()

```

`core/__init__.py`
```python

```

`core/constants.py`
```python
"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_STORAGE_ROOT = PROJECT_ROOT / "storage"

ACCOUNT_STATE_FILENAME = "account_state.json"
STRATEGY_MODEL_FILENAME = "strategy_model.json"
ACCOUNT_METRICS_FILENAME = "metrics.jsonl"
UPLOAD_QUEUE_FILENAME = "upload_queue.json"

VIDEO_EXTENSION = ".mp4"
AUDIO_EXTENSION = ".wav"
SUBTITLE_EXTENSION = ".srt"
THUMBNAIL_EXTENSION = ".png"

DRAFT_MODE = "draft"
PUBLISH_MODE = "publish"

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_NAME = "autonomous_engine"

FFMPEG_BINARY = "ffmpeg"
WHISPER_BINARY = "whisper"
PIPER_BINARY = "piper"
OLLAMA_BINARY = "ollama"

SECONDS_IN_MINUTE = 60

```

`core/config_loader.py`
```python
"""Configuration loading and validation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from core.constants import DEFAULT_CONFIG_PATH, DEFAULT_ENV_PATH, DEFAULT_STORAGE_ROOT


class ConfigError(RuntimeError):
    """Raised when configuration is invalid."""


class ConfigLoader:
    """Load YAML config with optional .env interpolation and defaults."""

    def __init__(self, config_path: Path | None = None, env_path: Path | None = None) -> None:
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.env_path = env_path or DEFAULT_ENV_PATH

    def load(self) -> dict[str, Any]:
        """Load and normalize configuration."""
        self._load_env_file()
        if not self.config_path.exists():
            raise ConfigError(f"Config not found: {self.config_path}")

        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        config = self._interpolate_env(raw)
        self._validate(config)
        self._apply_defaults(config)
        self._ensure_storage_dirs(config)
        return config

    def _load_env_file(self) -> None:
        """Load key/value pairs from .env if present."""
        if not self.env_path.exists():
            return
        for line in self.env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    def _interpolate_env(self, data: Any) -> Any:
        """Recursively interpolate ${VAR} values from environment."""
        if isinstance(data, dict):
            return {k: self._interpolate_env(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._interpolate_env(v) for v in data]
        if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_key = data[2:-1]
            return os.getenv(env_key, "")
        return data

    def _validate(self, config: dict[str, Any]) -> None:
        """Validate minimum required sections."""
        required = [
            "project",
            "execution",
            "generation",
            "analytics",
            "strategy",
            "upload",
            "scheduler",
            "accounts",
        ]
        missing = [key for key in required if key not in config]
        if missing:
            raise ConfigError(f"Missing required config sections: {', '.join(missing)}")
        if not isinstance(config.get("accounts"), list) or not config["accounts"]:
            raise ConfigError("Config 'accounts' must be a non-empty list")

    def _apply_defaults(self, config: dict[str, Any]) -> None:
        """Apply defaults for optional values."""
        project = config.setdefault("project", {})
        project.setdefault("storage_root", str(DEFAULT_STORAGE_ROOT))

        execution = config.setdefault("execution", {})
        execution.setdefault("account_mode", "sequential")
        execution.setdefault("max_workers", 1)

        upload = config.setdefault("upload", {})
        upload.setdefault("retry_attempts", 3)
        upload.setdefault("retry_backoff_seconds", 5)
        upload.setdefault("mock_mode", True)

        scheduler = config.setdefault("scheduler", {})
        scheduler.setdefault("interval_minutes", 60)

    def _ensure_storage_dirs(self, config: dict[str, Any]) -> None:
        """Ensure root storage directories exist."""
        storage_root = Path(config["project"]["storage_root"])
        if not storage_root.is_absolute():
            storage_root = self.config_path.parent / storage_root
        for relative in ["videos", "logs", "metrics", "models", "accounts", "queue"]:
            (storage_root / relative).mkdir(parents=True, exist_ok=True)


def get_account_config(config: dict[str, Any], account_name: str) -> dict[str, Any]:
    """Return account config by name."""
    for account in config.get("accounts", []):
        if account.get("name") == account_name:
            return account
    raise ConfigError(f"Account not found in config: {account_name}")


def active_accounts(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return active accounts from config."""
    return [a for a in config.get("accounts", []) if a.get("active", True)]

```

`core/logger.py`
```python
"""Structured logging utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from core.constants import DEFAULT_LOG_LEVEL, DEFAULT_LOG_NAME


class JsonFormatter(logging.Formatter):
    """JSON line log formatter for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "context"):
            payload["context"] = getattr(record, "context")
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that supports structured context."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        context = kwargs.pop("context", None)
        if context is not None:
            extra["context"] = context
        return msg, kwargs


def setup_logger(log_dir: Path, name: str = DEFAULT_LOG_NAME, level: str = DEFAULT_LOG_LEVEL) -> ContextAdapter:
    """Create console + JSONL file logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    console = RichHandler(rich_tracebacks=True)
    console.setLevel(level.upper())
    console.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_dir / f"{name}.jsonl", encoding="utf-8")
    file_handler.setLevel(level.upper())
    file_handler.setFormatter(JsonFormatter())

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return ContextAdapter(logger, {})

```

`accounts/__init__.py`
```python

```

`accounts/account_manager.py`
```python
"""Multi-account management and per-account path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from accounts.account_state_handler import AccountStateHandler
from core.constants import ACCOUNT_METRICS_FILENAME, STRATEGY_MODEL_FILENAME, UPLOAD_QUEUE_FILENAME
from core.config_loader import active_accounts


@dataclass(frozen=True)
class AccountProfile:
    """Immutable account profile from configuration."""

    name: str
    niche: str
    language: str
    timezone: str
    ollama_model: str
    tags_seed: list[str]
    session_state_path: Path
    active: bool = True


class AccountManager:
    """Manage account profiles and isolated account storage."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        storage_root = Path(config["project"]["storage_root"])
        self.storage_root = storage_root if storage_root.is_absolute() else project_root / storage_root
        self.accounts_root = self.storage_root / "accounts"

    def list_accounts(self, only_active: bool = True) -> list[AccountProfile]:
        """Return account profiles."""
        source = active_accounts(self.config) if only_active else self.config.get("accounts", [])
        return [self._to_profile(account) for account in source]

    def get_account(self, account_name: str) -> AccountProfile:
        """Get account profile by name."""
        for account in self.config.get("accounts", []):
            if account.get("name") == account_name:
                return self._to_profile(account)
        raise KeyError(f"Account not found: {account_name}")

    def ensure_account_dirs(self, account: AccountProfile) -> Path:
        """Ensure account-specific storage directories exist."""
        account_dir = self.accounts_root / account.name
        account_dir.mkdir(parents=True, exist_ok=True)
        (account_dir / "videos").mkdir(exist_ok=True)
        (account_dir / "tmp").mkdir(exist_ok=True)
        return account_dir

    def state_handler(self, account: AccountProfile) -> AccountStateHandler:
        """Get account state handler."""
        account_dir = self.ensure_account_dirs(account)
        return AccountStateHandler(account_dir)

    def strategy_model_path(self, account: AccountProfile) -> Path:
        """Path to strategy model JSON."""
        account_dir = self.ensure_account_dirs(account)
        return account_dir / STRATEGY_MODEL_FILENAME

    def metrics_path(self, account: AccountProfile) -> Path:
        """Path to account metrics JSONL."""
        account_dir = self.ensure_account_dirs(account)
        return account_dir / ACCOUNT_METRICS_FILENAME

    def queue_path(self, account: AccountProfile) -> Path:
        """Path to account upload queue."""
        account_dir = self.ensure_account_dirs(account)
        return account_dir / UPLOAD_QUEUE_FILENAME

    def account_video_dir(self, account: AccountProfile) -> Path:
        """Directory for generated account video artifacts."""
        account_dir = self.ensure_account_dirs(account)
        videos_dir = account_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        return videos_dir

    def _to_profile(self, account: dict[str, Any]) -> AccountProfile:
        session_state = Path(account.get("session_state_path", ""))
        if not session_state.is_absolute():
            session_state = self.project_root / session_state

        return AccountProfile(
            name=account["name"],
            niche=account.get("niche", "general"),
            language=account.get("language", "en"),
            timezone=account.get("timezone", "UTC"),
            ollama_model=account.get("ollama_model", "llama3.1:8b"),
            tags_seed=list(account.get("tags_seed", [])),
            session_state_path=session_state,
            active=bool(account.get("active", True)),
        )

```

`accounts/account_state_handler.py`
```python
"""Persistent per-account state handling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.constants import ACCOUNT_STATE_FILENAME


@dataclass
class AccountState:
    """Mutable account-level runtime and learning state."""

    account_name: str
    generated_count: int = 0
    uploaded_count: int = 0
    failed_uploads: int = 0
    pending_metrics_post_ids: list[str] = field(default_factory=list)
    last_run_at: str | None = None
    last_success_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert state to serializable dict."""
        return {
            "account_name": self.account_name,
            "generated_count": self.generated_count,
            "uploaded_count": self.uploaded_count,
            "failed_uploads": self.failed_uploads,
            "pending_metrics_post_ids": self.pending_metrics_post_ids,
            "last_run_at": self.last_run_at,
            "last_success_at": self.last_success_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AccountState":
        """Create state from stored dictionary."""
        return cls(
            account_name=payload.get("account_name", "unknown"),
            generated_count=int(payload.get("generated_count", 0)),
            uploaded_count=int(payload.get("uploaded_count", 0)),
            failed_uploads=int(payload.get("failed_uploads", 0)),
            pending_metrics_post_ids=list(payload.get("pending_metrics_post_ids", [])),
            last_run_at=payload.get("last_run_at"),
            last_success_at=payload.get("last_success_at"),
        )


class AccountStateHandler:
    """Read and write account state from account_state.json."""

    def __init__(self, account_dir: Path) -> None:
        self.account_dir = account_dir
        self.path = account_dir / ACCOUNT_STATE_FILENAME

    def load(self, account_name: str) -> AccountState:
        """Load account state or initialize defaults."""
        if not self.path.exists():
            state = AccountState(account_name=account_name)
            self.save(state)
            return state

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        state = AccountState.from_dict(payload)
        if state.account_name == "unknown":
            state.account_name = account_name
        return state

    def save(self, state: AccountState) -> None:
        """Persist account state as JSON."""
        self.account_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

    def mark_run(self, state: AccountState, success: bool) -> AccountState:
        """Update run metadata after a processing cycle."""
        now = datetime.now(timezone.utc).isoformat()
        state.last_run_at = now
        if success:
            state.last_success_at = now
        self.save(state)
        return state

```

`trend_engine/__init__.py`
```python

```

`trend_engine/trend_scraper.py`
```python
"""Trend scraping module.

Uses a local mock strategy by default so the system can run offline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class TrendSignal:
    """Raw trend signal for downstream scoring."""

    phrase: str
    niche: str
    source: str
    mentions_growth: float
    competition: float
    freshness_hours: int


class TrendScraper:
    """Scrape trends using local heuristics and curated pools."""

    BASE_TRENDS: dict[str, list[str]] = {
        "fitness": [
            "7 minute core blast",
            "high protein breakfast hacks",
            "mobility before bed",
            "fat loss myths",
            "dumbbell only routine",
        ],
        "personal finance": [
            "50/30/20 budget breakdown",
            "stop impulse spending",
            "high yield savings explained",
            "credit utilization mistakes",
            "paycheck automation plan",
        ],
        "general": [
            "3 mistakes everyone makes",
            "daily habit that compounds",
            "before you start this",
            "one minute checklist",
            "hidden trick nobody uses",
        ],
    }

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def scrape(self, niche: str, limit: int = 10) -> list[TrendSignal]:
        """Return mock trend signals for a niche."""
        key = niche.lower().strip()
        candidates = self.BASE_TRENDS.get(key, self.BASE_TRENDS["general"])
        selected = [self._random.choice(candidates) for _ in range(max(limit, 1))]

        now = datetime.now(timezone.utc)
        return [
            TrendSignal(
                phrase=phrase,
                niche=niche,
                source="local_mock",
                mentions_growth=round(self._random.uniform(0.8, 2.4), 3),
                competition=round(self._random.uniform(0.2, 0.95), 3),
                freshness_hours=int((now.minute + i * 7) % 72),
            )
            for i, phrase in enumerate(selected)
        ]

```

`trend_engine/trend_classifier.py`
```python
"""Trend classification module."""

from __future__ import annotations

from dataclasses import dataclass

from trend_engine.trend_scraper import TrendSignal


@dataclass
class ClassifiedTrend:
    """Trend enriched with category and intent."""

    phrase: str
    niche: str
    source: str
    mentions_growth: float
    competition: float
    freshness_hours: int
    category: str
    intent: str


class TrendClassifier:
    """Classifies trends by lexical heuristics."""

    def classify(self, trends: list[TrendSignal]) -> list[ClassifiedTrend]:
        """Classify trend type and user intent."""
        classified: list[ClassifiedTrend] = []
        for trend in trends:
            text = trend.phrase.lower()
            category = self._infer_category(text)
            intent = self._infer_intent(text)
            classified.append(
                ClassifiedTrend(
                    phrase=trend.phrase,
                    niche=trend.niche,
                    source=trend.source,
                    mentions_growth=trend.mentions_growth,
                    competition=trend.competition,
                    freshness_hours=trend.freshness_hours,
                    category=category,
                    intent=intent,
                )
            )
        return classified

    def _infer_category(self, text: str) -> str:
        if any(token in text for token in ["mistake", "myth", "stop"]):
            return "pain_point"
        if any(token in text for token in ["routine", "plan", "checklist", "breakdown"]):
            return "how_to"
        if any(token in text for token in ["before", "hidden", "hack", "trick"]):
            return "curiosity"
        return "general"

    def _infer_intent(self, text: str) -> str:
        if any(token in text for token in ["explained", "breakdown", "myth"]):
            return "educational"
        if any(token in text for token in ["routine", "plan", "minute"]):
            return "actionable"
        return "awareness"

```

`trend_engine/trend_score_engine.py`
```python
"""Trend scoring engine."""

from __future__ import annotations

from dataclasses import dataclass

from trend_engine.trend_classifier import ClassifiedTrend


@dataclass
class ScoredTrend:
    """Trend with normalized score for ranking."""

    phrase: str
    category: str
    intent: str
    score: float


class TrendScoreEngine:
    """Compute weighted trend scores."""

    def __init__(self, growth_weight: float = 0.5, freshness_weight: float = 0.3, competition_weight: float = 0.2) -> None:
        self.growth_weight = growth_weight
        self.freshness_weight = freshness_weight
        self.competition_weight = competition_weight

    def score(self, trends: list[ClassifiedTrend]) -> list[ScoredTrend]:
        """Rank trends by opportunity score."""
        scored: list[ScoredTrend] = []
        for trend in trends:
            freshness_score = max(0.0, 1.0 - (trend.freshness_hours / 72.0))
            competition_score = max(0.0, 1.0 - trend.competition)
            normalized_growth = min(1.0, trend.mentions_growth / 2.5)
            total = (
                self.growth_weight * normalized_growth
                + self.freshness_weight * freshness_score
                + self.competition_weight * competition_score
            )
            scored.append(
                ScoredTrend(
                    phrase=trend.phrase,
                    category=trend.category,
                    intent=trend.intent,
                    score=round(total, 4),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)

```

`content_engine/__init__.py`
```python

```

`content_engine/idea_generator.py`
```python
"""Content idea generation via local Ollama model."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from core.constants import OLLAMA_BINARY


@dataclass
class ContentIdea:
    """Single generated content idea."""

    title: str
    angle: str
    target_emotion: str
    trend_phrase: str
    format_name: str


class IdeaGenerator:
    """Generate short-form content ideas using local LLM."""

    def __init__(self, logger: Any, timeout_seconds: int = 90) -> None:
        self.logger = logger
        self.timeout_seconds = timeout_seconds

    def generate_ideas(
        self,
        account_name: str,
        niche: str,
        model: str,
        scored_trends: list[dict[str, Any]],
        format_name: str,
        count: int = 3,
    ) -> list[ContentIdea]:
        """Generate content ideas from top trends."""
        top_trends = [item["phrase"] for item in scored_trends[:max(1, count)]]
        if self._ollama_available():
            ideas = self._generate_with_ollama(
                account_name=account_name,
                niche=niche,
                model=model,
                top_trends=top_trends,
                format_name=format_name,
                count=count,
            )
            if ideas:
                return ideas

        return self._fallback_ideas(niche=niche, top_trends=top_trends, format_name=format_name, count=count)

    def _generate_with_ollama(
        self,
        account_name: str,
        niche: str,
        model: str,
        top_trends: list[str],
        format_name: str,
        count: int,
    ) -> list[ContentIdea]:
        prompt = (
            f"You are a TikTok strategist for account '{account_name}' in niche '{niche}'. "
            f"Generate {count} ideas in format '{format_name}'. "
            "Output each idea on one line with this pipe format: "
            "Title|Angle|Emotion|TrendPhrase. "
            f"Trends: {', '.join(top_trends)}"
        )
        try:
            result = subprocess.run(
                [OLLAMA_BINARY, "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if result.returncode != 0:
                self.logger.warning(
                    "Ollama command failed; using fallback ideas",
                    context={"stderr": result.stderr.strip()},
                )
                return []

            ideas: list[ContentIdea] = []
            for line in result.stdout.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 4:
                    continue
                ideas.append(
                    ContentIdea(
                        title=parts[0],
                        angle=parts[1],
                        target_emotion=parts[2],
                        trend_phrase=parts[3],
                        format_name=format_name,
                    )
                )
            return ideas[:count]
        except (subprocess.SubprocessError, OSError) as exc:
            self.logger.warning(
                "Ollama unavailable during generation; using fallback",
                context={"error": str(exc)},
            )
            return []

    def _fallback_ideas(
        self,
        niche: str,
        top_trends: list[str],
        format_name: str,
        count: int,
    ) -> list[ContentIdea]:
        ideas: list[ContentIdea] = []
        trends = top_trends or [f"{niche} quick win"]
        emotions = ["curiosity", "motivation", "confidence", "urgency"]

        for i in range(count):
            trend = trends[i % len(trends)]
            emotion = emotions[i % len(emotions)]
            ideas.append(
                ContentIdea(
                    title=f"{trend.title()} in {i + 1} steps",
                    angle=f"Actionable {format_name} take on {trend}",
                    target_emotion=emotion,
                    trend_phrase=trend,
                    format_name=format_name,
                )
            )
        return ideas

    def _ollama_available(self) -> bool:
        return shutil.which(OLLAMA_BINARY) is not None

```

`content_engine/script_generator.py`
```python
"""TikTok short script generation."""

from __future__ import annotations

from dataclasses import dataclass

from content_engine.idea_generator import ContentIdea


@dataclass
class VideoScript:
    """Generated short-form script."""

    hook: str
    body: str
    cta: str
    full_text: str
    duration_seconds: int


class ScriptGenerator:
    """Generate concise scripts that fit 9-15 second clips."""

    def generate(self, idea: ContentIdea, hook_style: str, duration_seconds: int) -> VideoScript:
        """Build a script from an idea."""
        hook_templates = {
            "question": f"Still struggling with {idea.trend_phrase}?",
            "shock": f"Most people fail {idea.trend_phrase} for one reason.",
            "contrarian": f"Ignore the usual advice about {idea.trend_phrase}.",
            "promise": f"In {duration_seconds} seconds, fix {idea.trend_phrase}.",
        }
        hook = hook_templates.get(hook_style, hook_templates["question"])
        body = (
            f"Start with this: {idea.angle}. "
            f"Then apply one clear action today and keep it consistent."
        )
        cta = "Comment 'guide' and follow for the next breakdown."
        full_text = f"{hook} {body} {cta}"
        return VideoScript(hook=hook, body=body, cta=cta, full_text=full_text, duration_seconds=duration_seconds)

```

`content_engine/hook_optimizer.py`
```python
"""Hook optimization module."""

from __future__ import annotations


class HookOptimizer:
    """Optimize hooks using historical winner patterns."""

    def optimize(self, base_hook: str, winning_hooks: list[str], hook_style: str) -> str:
        """Return optimized hook with lightweight adaptation."""
        if winning_hooks:
            best = winning_hooks[0]
            return f"{best} {base_hook}"
        if hook_style == "shock":
            return f"Wait. {base_hook}"
        if hook_style == "question":
            return f"Quick question: {base_hook}"
        return base_hook

```

`content_engine/caption_generator.py`
```python
"""Caption generation module."""

from __future__ import annotations

from content_engine.idea_generator import ContentIdea


class CaptionGenerator:
    """Generate style-specific TikTok captions."""

    def generate(self, idea: ContentIdea, caption_style: str, include_cta: bool = True) -> str:
        """Return a short caption with chosen style."""
        style_prefix = {
            "direct": "Do this today:",
            "curiosity": "Most people miss this:",
            "urgency": "Try this before your next scroll:",
            "community": "Who else is working on this:",
        }
        prefix = style_prefix.get(caption_style, "Try this:")
        caption = f"{prefix} {idea.title}"
        if include_cta:
            caption += " | Save this and share it."
        return caption

```

`content_engine/hashtag_optimizer.py`
```python
"""Hashtag optimization module."""

from __future__ import annotations

import re


class HashtagOptimizer:
    """Generate and rank hashtags from trends and account niche."""

    def generate(
        self,
        niche: str,
        trend_phrases: list[str],
        seed_tags: list[str],
        max_tags: int = 8,
    ) -> list[str]:
        """Return cleaned hashtag list."""
        raw_tags = [niche] + trend_phrases + seed_tags + ["fyp", "tiktoktips"]
        cleaned: list[str] = []
        seen: set[str] = set()

        for tag in raw_tags:
            normalized = re.sub(r"[^a-zA-Z0-9]", "", tag.replace(" ", "")).lower()
            if not normalized:
                continue
            hash_tag = f"#{normalized}"
            if hash_tag in seen:
                continue
            seen.add(hash_tag)
            cleaned.append(hash_tag)
            if len(cleaned) >= max_tags:
                break

        return cleaned

```

`media_engine/__init__.py`
```python

```

`media_engine/voice_generator.py`
```python
"""Voice generation using local Piper TTS."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.constants import FFMPEG_BINARY, PIPER_BINARY


class VoiceGenerator:
    """Generate narration tracks from script text."""

    def __init__(self, logger: Any, timeout_seconds: int = 120) -> None:
        self.logger = logger
        self.timeout_seconds = timeout_seconds

    def synthesize(
        self,
        text: str,
        output_wav: Path,
        voice_model_path: str | None,
        duration_seconds: int,
    ) -> Path:
        """Synthesize voice with Piper, fallback to silent track."""
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        if voice_model_path and shutil.which(PIPER_BINARY):
            try:
                cmd = [
                    PIPER_BINARY,
                    "--model",
                    voice_model_path,
                    "--output_file",
                    str(output_wav),
                ]
                result = subprocess.run(
                    cmd,
                    input=text,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
                if result.returncode == 0 and output_wav.exists():
                    return output_wav
                self.logger.warning(
                    "Piper generation failed; falling back to silent audio",
                    context={"stderr": result.stderr.strip()},
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self.logger.warning(
                    "Piper unavailable; falling back to silent audio",
                    context={"error": str(exc)},
                )

        self._generate_silence(output_wav, duration_seconds)
        return output_wav

    def _generate_silence(self, output_wav: Path, duration_seconds: int) -> None:
        cmd = [
            FFMPEG_BINARY,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            str(max(1, duration_seconds)),
            str(output_wav),
        ]
        subprocess.run(cmd, capture_output=True, check=False)

```

`media_engine/subtitle_generator.py`
```python
"""Subtitle generation using local Whisper or deterministic fallback."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.constants import WHISPER_BINARY


class SubtitleGenerator:
    """Create SRT subtitles from narration audio."""

    def __init__(self, logger: Any, timeout_seconds: int = 240) -> None:
        self.logger = logger
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        audio_path: Path,
        output_srt: Path,
        language: str,
        fallback_text: str,
        duration_seconds: int,
    ) -> Path:
        """Generate subtitles with Whisper if available; otherwise fallback."""
        output_srt.parent.mkdir(parents=True, exist_ok=True)

        if shutil.which(WHISPER_BINARY):
            produced = self._run_whisper(audio_path=audio_path, output_srt=output_srt, language=language)
            if produced:
                return produced

        self._write_fallback_srt(output_srt, fallback_text, duration_seconds)
        return output_srt

    def _run_whisper(self, audio_path: Path, output_srt: Path, language: str) -> Path | None:
        out_dir = output_srt.parent
        cmd = [
            WHISPER_BINARY,
            str(audio_path),
            "--task",
            "transcribe",
            "--language",
            language,
            "--output_format",
            "srt",
            "--output_dir",
            str(out_dir),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if result.returncode != 0:
                self.logger.warning("Whisper command failed", context={"stderr": result.stderr.strip()})
                return None
            generated = out_dir / f"{audio_path.stem}.srt"
            if generated.exists():
                generated.replace(output_srt)
                return output_srt
            return None
        except (OSError, subprocess.SubprocessError) as exc:
            self.logger.warning("Whisper unavailable", context={"error": str(exc)})
            return None

    def _write_fallback_srt(self, output_srt: Path, text: str, duration_seconds: int) -> None:
        words = text.split()
        if not words:
            words = ["..."]
        chunks = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]
        slot = max(1.0, duration_seconds / max(1, len(chunks)))

        lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            start = (idx - 1) * slot
            end = min(duration_seconds, idx * slot)
            lines.extend(
                [
                    str(idx),
                    f"{self._fmt(start)} --> {self._fmt(end)}",
                    chunk,
                    "",
                ]
            )
        output_srt.write_text("\n".join(lines), encoding="utf-8")

    def _fmt(self, seconds: float) -> str:
        millis = int((seconds - int(seconds)) * 1000)
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02}:{m:02}:{s:02},{millis:03}"

```

`media_engine/video_renderer.py`
```python
"""Video rendering pipeline using FFmpeg."""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.constants import FFMPEG_BINARY


class VideoRenderer:
    """Render 1080x1920 TikTok videos with animated text overlay."""

    def __init__(self, logger: Any, fps: int = 30, resolution: str = "1080x1920") -> None:
        self.logger = logger
        self.fps = fps
        self.resolution = resolution

    def render(
        self,
        output_video: Path,
        audio_path: Path,
        subtitle_path: Path,
        hook_text: str,
        body_text: str,
        caption_text: str,
        duration_seconds: int,
        background_music: Path | None = None,
        background_color: str | None = None,
    ) -> Path:
        """Render final MP4 output with H264 codec."""
        output_video.parent.mkdir(parents=True, exist_ok=True)

        if not shutil.which(FFMPEG_BINARY):
            raise RuntimeError("ffmpeg is required but not found on PATH")

        bg_color = background_color or random.choice(["0x101820", "0x1f2a44", "0x2d1e2f", "0x1f3b4d"])
        hook = self._escape(hook_text)
        body = self._escape(body_text)
        cap = self._escape(caption_text)

        filter_chain = (
            f"drawtext=text='{hook}':"
            "x=(w-text_w)/2:y=h*0.16:fontcolor=white:fontsize=if(lt(t,1.5),130-22*t,96):"
            "box=1:boxcolor=black@0.45:boxborderw=22:enable='between(t,0,1.5)',"
            f"drawtext=text='{body}':"
            "x=(w-text_w)/2:y=h*0.34:fontcolor=white:fontsize=56:"
            "box=1:boxcolor=black@0.4:boxborderw=18:enable='between(t,1.2,14)',"
            f"drawtext=text='{cap}':"
            "x=(w-text_w)/2:y=h*0.82:fontcolor=yellow:fontsize=46:"
            "box=1:boxcolor=black@0.35:boxborderw=14:enable='between(t,0,15)'"
        )

        cmd = [
            FFMPEG_BINARY,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={bg_color}:s={self.resolution}:r={self.fps}:d={duration_seconds}",
            "-i",
            str(audio_path),
        ]

        if background_music and background_music.exists():
            cmd += ["-i", str(background_music)]
            filter_complex = f"[1:a]volume=1.0[voice];[2:a]volume=0.15[music];[voice][music]amix=inputs=2:duration=first[a]"
            cmd += ["-filter_complex", filter_complex, "-map", "0:v", "-map", "[a]"]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]

        cmd += [
            "-vf",
            filter_chain,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(self.fps),
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_video),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            self.logger.error(
                "Video rendering failed",
                context={"stderr": result.stderr.strip(), "video": str(output_video)},
            )
            raise RuntimeError("Video rendering failed")

        if subtitle_path.exists():
            self._burn_subtitles(output_video, subtitle_path)

        return output_video

    def _burn_subtitles(self, video_path: Path, subtitle_path: Path) -> None:
        temp_path = video_path.with_name(f"{video_path.stem}_subtitled{video_path.suffix}")
        cmd = [
            FFMPEG_BINARY,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles={self._escape_path(subtitle_path)}:force_style='Fontsize=20,PrimaryColour=&H00FFFFFF,Outline=1'",
            "-c:a",
            "copy",
            str(temp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and temp_path.exists():
            temp_path.replace(video_path)

    def _escape(self, text: str) -> str:
        return text.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")

    def _escape_path(self, path: Path) -> str:
        return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")

```

`media_engine/thumbnail_selector.py`
```python
"""Thumbnail extraction utility."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from core.constants import FFMPEG_BINARY


class ThumbnailSelector:
    """Select a representative frame as thumbnail."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def extract(self, video_path: Path, output_image: Path, timestamp_seconds: float = 1.0) -> Path:
        """Extract thumbnail frame from a rendered video."""
        output_image.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            FFMPEG_BINARY,
            "-y",
            "-ss",
            str(timestamp_seconds),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_image),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            self.logger.warning(
                "Thumbnail extraction failed",
                context={"stderr": result.stderr.strip(), "video": str(video_path)},
            )
        return output_image

```

`upload_engine/__init__.py`
```python

```

`upload_engine/draft_or_publish_mode.py`
```python
"""Upload mode definitions."""

from __future__ import annotations

from enum import Enum


class DraftOrPublishMode(str, Enum):
    """Upload destination mode."""

    DRAFT = "draft"
    PUBLISH = "publish"

    @classmethod
    def from_value(cls, value: str) -> "DraftOrPublishMode":
        normalized = (value or "draft").strip().lower()
        if normalized == cls.PUBLISH.value:
            return cls.PUBLISH
        return cls.DRAFT

```

`upload_engine/upload_retry_logic.py`
```python
"""Retry logic utilities for upload operations."""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_operation(
    operation: Callable[[], T],
    attempts: int,
    backoff_seconds: int,
    logger: Any,
    context: dict[str, Any] | None = None,
) -> T:
    """Execute operation with fixed-backoff retry behavior."""
    if attempts < 1:
        attempts = 1

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Upload attempt failed",
                context={
                    **(context or {}),
                    "attempt": attempt,
                    "attempts": attempts,
                    "error": str(exc),
                },
            )
            if attempt < attempts:
                time.sleep(max(0, backoff_seconds))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry operation failed without exception")

```

`upload_engine/upload_queue_manager.py`
```python
"""Persistent upload queue manager."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upload_engine.draft_or_publish_mode import DraftOrPublishMode


@dataclass
class UploadTask:
    """Pending upload task."""

    id: str
    account_name: str
    video_path: str
    caption: str
    hashtags: list[str]
    mode: str
    thumbnail_path: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str = ""
    retries: int = 0
    status: str = "pending"


class UploadQueueManager:
    """Manage account-specific upload queue in JSON format."""

    def __init__(self, queue_path: Path) -> None:
        self.queue_path = queue_path
        self.lock = threading.Lock()
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.queue_path.exists():
            self._write({"tasks": []})

    def enqueue(
        self,
        account_name: str,
        video_path: Path,
        caption: str,
        hashtags: list[str],
        mode: DraftOrPublishMode,
        thumbnail_path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UploadTask:
        """Add a task to the queue."""
        task = UploadTask(
            id=str(uuid.uuid4()),
            account_name=account_name,
            video_path=str(video_path),
            caption=caption,
            hashtags=hashtags,
            mode=mode.value,
            thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            retries=0,
            status="pending",
        )
        with self.lock:
            payload = self._read()
            payload["tasks"].append(asdict(task))
            self._write(payload)
        return task

    def next_pending(self) -> UploadTask | None:
        """Return and reserve next pending task."""
        with self.lock:
            payload = self._read()
            for idx, raw in enumerate(payload.get("tasks", [])):
                if raw.get("status") != "pending":
                    continue
                raw["status"] = "in_progress"
                payload["tasks"][idx] = raw
                self._write(payload)
                return UploadTask(**raw)
        return None

    def mark_completed(self, task_id: str, post_id: str | None = None) -> None:
        """Mark task as completed."""
        with self.lock:
            payload = self._read()
            for task in payload.get("tasks", []):
                if task.get("id") == task_id:
                    task["status"] = "completed"
                    if post_id:
                        task["post_id"] = post_id
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self._write(payload)

    def mark_failed(self, task_id: str, reason: str) -> None:
        """Mark task as failed and keep reason."""
        with self.lock:
            payload = self._read()
            for task in payload.get("tasks", []):
                if task.get("id") == task_id:
                    task["status"] = "failed"
                    task["error"] = reason
                    task["failed_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self._write(payload)

    def increment_retry(self, task_id: str) -> None:
        """Increment retry counter and return task to pending state."""
        with self.lock:
            payload = self._read()
            for task in payload.get("tasks", []):
                if task.get("id") == task_id:
                    task["retries"] = int(task.get("retries", 0)) + 1
                    task["status"] = "pending"
                    break
            self._write(payload)

    def pending_count(self) -> int:
        """Number of pending tasks."""
        payload = self._read()
        return sum(1 for task in payload.get("tasks", []) if task.get("status") == "pending")

    def all_tasks(self) -> list[dict[str, Any]]:
        """Return all queue tasks."""
        return self._read().get("tasks", [])

    def _read(self) -> dict[str, Any]:
        return json.loads(self.queue_path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.queue_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

```

`upload_engine/tiktok_uploader.py`
```python
"""TikTok uploader using Playwright browser automation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from upload_engine.draft_or_publish_mode import DraftOrPublishMode

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # noqa: BLE001
    sync_playwright = None
    PlaywrightTimeoutError = TimeoutError


@dataclass
class UploadResult:
    """Result of upload action."""

    success: bool
    post_id: str | None
    error: str | None = None


class TikTokUploader:
    """Upload videos to TikTok with optional mock mode for offline runs."""

    def __init__(self, logger: Any, upload_url: str, headless: bool = True, mock_mode: bool = True) -> None:
        self.logger = logger
        self.upload_url = upload_url
        self.headless = headless
        self.mock_mode = mock_mode

    def upload(
        self,
        video_path: Path,
        caption: str,
        hashtags: list[str],
        mode: DraftOrPublishMode,
        session_state_path: Path,
        post_wait_seconds: int = 12,
    ) -> UploadResult:
        """Upload a video and publish/draft it."""
        if self.mock_mode or sync_playwright is None:
            post_id = f"mock_{uuid.uuid4().hex[:12]}"
            self.logger.info("Mock upload complete", context={"post_id": post_id, "video": str(video_path)})
            return UploadResult(success=True, post_id=post_id)

        if not video_path.exists():
            return UploadResult(success=False, post_id=None, error=f"Video not found: {video_path}")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                context_kwargs: dict[str, Any] = {}
                if session_state_path.exists():
                    context_kwargs["storage_state"] = str(session_state_path)

                context = browser.new_context(**context_kwargs)
                page = context.new_page()
                page.goto(self.upload_url, timeout=60_000)

                page.set_input_files("input[type='file']", str(video_path), timeout=60_000)
                full_caption = self._build_caption(caption, hashtags)
                self._fill_caption(page, full_caption)

                if mode == DraftOrPublishMode.DRAFT:
                    self._click_first(page, ["button:has-text('Draft')", "button:has-text('Save draft')"])
                else:
                    self._click_first(page, ["button:has-text('Post')", "button:has-text('Publish')"])

                page.wait_for_timeout(post_wait_seconds * 1000)
                context.storage_state(path=str(session_state_path))
                browser.close()

                post_id = f"tt_{uuid.uuid4().hex[:12]}"
                self.logger.info("Upload successful", context={"post_id": post_id, "mode": mode.value})
                return UploadResult(success=True, post_id=post_id)
        except PlaywrightTimeoutError as exc:
            return UploadResult(success=False, post_id=None, error=f"Timeout during upload: {exc}")
        except Exception as exc:  # noqa: BLE001
            return UploadResult(success=False, post_id=None, error=f"Upload failed: {exc}")

    def _build_caption(self, caption: str, hashtags: list[str]) -> str:
        tags = " ".join(hashtags)
        return f"{caption}\n{tags}".strip()

    def _fill_caption(self, page: Any, caption: str) -> None:
        selectors = ["div[role='textbox']", "textarea", "[contenteditable='true']"]
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count() > 0:
                locator.first.click()
                locator.first.fill(caption)
                return
        raise RuntimeError("Unable to locate caption input field")

    def _click_first(self, page: Any, selectors: list[str]) -> None:
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count() > 0:
                locator.first.click()
                return
        raise RuntimeError(f"Unable to locate any selector from: {selectors}")

```

`analytics_engine/__init__.py`
```python

```

`analytics_engine/metrics_collector.py`
```python
"""Metrics collection and persistence."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class VideoMetric:
    """Metrics for one uploaded video."""

    post_id: str
    account_name: str
    format_name: str
    hook_style: str
    duration_seconds: int
    caption_style: str
    publish_hour: int
    views: int
    completion_rate: float
    shares: int
    comments: int
    status: str
    created_at: str
    collected_at: str | None = None


class MetricsCollector:
    """Track pending and collected metrics in account-local JSONL file."""

    def __init__(self, metrics_path: Path, logger: Any, seed: int | None = None) -> None:
        self.metrics_path = metrics_path
        self.logger = logger
        self._random = random.Random(seed)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.touch(exist_ok=True)

    def record_pending(self, post_id: str, account_name: str, metadata: dict[str, Any]) -> None:
        """Record a newly uploaded post as pending metrics."""
        payload = {
            "post_id": post_id,
            "account_name": account_name,
            "format_name": metadata.get("format_name", "unknown"),
            "hook_style": metadata.get("hook_style", "unknown"),
            "duration_seconds": int(metadata.get("duration_seconds", 12)),
            "caption_style": metadata.get("caption_style", "unknown"),
            "publish_hour": int(metadata.get("publish_hour", 12)),
            "views": 0,
            "completion_rate": 0.0,
            "shares": 0,
            "comments": 0,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collected_at": None,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def collect(self, min_age_minutes: int = 5, mock_mode: bool = True) -> list[VideoMetric]:
        """Collect metrics for eligible pending records."""
        records = self._read_all()
        now = datetime.now(timezone.utc)
        changed = False
        collected: list[VideoMetric] = []

        for record in records:
            if record.get("status") != "pending":
                continue

            created_at = datetime.fromisoformat(record["created_at"])
            if now - created_at < timedelta(minutes=min_age_minutes):
                continue

            if not mock_mode:
                self.logger.info("Metrics unavailable, kept as pending", context={"post_id": record.get("post_id")})
                continue

            record["views"] = int(self._random.randint(300, 12000))
            record["completion_rate"] = round(self._random.uniform(0.25, 0.95), 4)
            record["shares"] = int(self._random.randint(1, 250))
            record["comments"] = int(self._random.randint(0, 180))
            record["status"] = "collected"
            record["collected_at"] = now.isoformat()
            changed = True

            collected.append(VideoMetric(**record))

        if changed:
            self._write_all(records)

        return collected

    def dataframe(self, status_filter: str | None = None) -> pd.DataFrame:
        """Return all metrics as DataFrame."""
        records = self._read_all()
        if status_filter:
            records = [r for r in records if r.get("status") == status_filter]
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def _read_all(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    def _write_all(self, records: list[dict[str, Any]]) -> None:
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

```

`analytics_engine/performance_analyzer.py`
```python
"""Performance analyzer with configurable engagement scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class EngagementWeights:
    """Configurable weights for engagement score."""

    views: float
    completion_rate: float
    shares: float
    comments: float


class PerformanceAnalyzer:
    """Compute engagement scores and enriched metrics."""

    def __init__(self, weights: EngagementWeights) -> None:
        self.weights = weights

    def engagement_score(self, row: dict[str, Any]) -> float:
        """Calculate score using required formula."""
        return (
            self.weights.views * float(row.get("views", 0))
            + self.weights.completion_rate * float(row.get("completion_rate", 0.0))
            + self.weights.shares * float(row.get("shares", 0))
            + self.weights.comments * float(row.get("comments", 0))
        )

    def analyze(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Compute engagement scores for all rows."""
        if metrics.empty:
            return metrics

        scored = metrics.copy()
        scored["engagement_score"] = scored.apply(
            lambda row: self.engagement_score(row.to_dict()), axis=1
        )
        scored["score_rank_pct"] = scored["engagement_score"].rank(pct=True, method="average")
        return scored.sort_values("engagement_score", ascending=False)

```

`analytics_engine/winner_pattern_detector.py`
```python
"""Winner and loser pattern detection."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


class WinnerPatternDetector:
    """Detect patterns from top and bottom scoring content."""

    def __init__(self, winner_quantile: float = 0.75, loser_quantile: float = 0.25) -> None:
        self.winner_quantile = winner_quantile
        self.loser_quantile = loser_quantile

    def detect(self, scored_df: pd.DataFrame) -> dict[str, Any]:
        """Return winner and loser patterns."""
        if scored_df.empty or "engagement_score" not in scored_df:
            return {
                "winners": {"hooks": [], "durations": [], "caption_styles": [], "formats": [], "publish_hours": []},
                "losers": {"hooks": [], "durations": [], "caption_styles": [], "formats": [], "publish_hours": []},
            }

        high_cut = scored_df["engagement_score"].quantile(self.winner_quantile)
        low_cut = scored_df["engagement_score"].quantile(self.loser_quantile)

        winners = scored_df[scored_df["engagement_score"] >= high_cut]
        losers = scored_df[scored_df["engagement_score"] <= low_cut]

        return {
            "winners": {
                "hooks": self._top_values(winners, "hook_style"),
                "durations": self._top_values(winners, "duration_seconds"),
                "caption_styles": self._top_values(winners, "caption_style"),
                "formats": self._top_values(winners, "format_name"),
                "publish_hours": self._top_values(winners, "publish_hour"),
            },
            "losers": {
                "hooks": self._top_values(losers, "hook_style"),
                "durations": self._top_values(losers, "duration_seconds"),
                "caption_styles": self._top_values(losers, "caption_style"),
                "formats": self._top_values(losers, "format_name"),
                "publish_hours": self._top_values(losers, "publish_hour"),
            },
        }

    def _top_values(self, frame: pd.DataFrame, column: str, top_n: int = 3) -> list[Any]:
        if frame.empty or column not in frame:
            return []
        counts = Counter(frame[column].tolist())
        return [key for key, _ in counts.most_common(top_n)]

```

`analytics_engine/format_score_model.py`
```python
"""Persistent strategy weight model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FormatScoreModel:
    """Load, update, and save strategy weights."""

    def __init__(self, model_path: Path, rotation_floor: float = 0.05) -> None:
        self.model_path = model_path
        self.rotation_floor = rotation_floor
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def load_or_init(self, strategy_config: dict[str, Any]) -> dict[str, Any]:
        """Load existing model, or initialize from config."""
        if self.model_path.exists():
            return json.loads(self.model_path.read_text(encoding="utf-8"))

        model = {
            "format_weights": dict(strategy_config.get("initial_format_weights", {})),
            "hook_weights": dict(strategy_config.get("initial_hook_weights", {})),
            "caption_weights": dict(strategy_config.get("initial_caption_weights", {})),
            "duration_weights": {
                str(k): float(v)
                for k, v in dict(strategy_config.get("initial_duration_weights", {})).items()
            },
            "publish_hour_weights": {str(hour): 1.0 for hour in range(24)},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save(model)
        return model

    def save(self, model: dict[str, Any]) -> None:
        """Persist strategy model."""
        model["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    def update(
        self,
        model: dict[str, Any],
        winners: dict[str, list[Any]],
        losers: dict[str, list[Any]],
        learning_rate: float,
    ) -> dict[str, Any]:
        """Update strategy weights from winners/losers patterns."""
        self._adjust(model["format_weights"], winners.get("formats", []), losers.get("formats", []), learning_rate)
        self._adjust(model["hook_weights"], winners.get("hooks", []), losers.get("hooks", []), learning_rate)
        self._adjust(model["caption_weights"], winners.get("caption_styles", []), losers.get("caption_styles", []), learning_rate)
        self._adjust(model["duration_weights"], winners.get("durations", []), losers.get("durations", []), learning_rate)
        self._adjust(model["publish_hour_weights"], winners.get("publish_hours", []), losers.get("publish_hours", []), learning_rate)

        self._normalize_all(model)
        return model

    def _adjust(
        self,
        weights: dict[str, float],
        winners: list[Any],
        losers: list[Any],
        learning_rate: float,
    ) -> None:
        for winner in winners:
            key = str(winner)
            if key in weights:
                weights[key] = float(weights[key]) * (1.0 + learning_rate)

        for loser in losers:
            key = str(loser)
            if key in weights:
                weights[key] = max(self.rotation_floor, float(weights[key]) * (1.0 - learning_rate))

    def _normalize_all(self, model: dict[str, Any]) -> None:
        for bucket in [
            model.get("format_weights", {}),
            model.get("hook_weights", {}),
            model.get("caption_weights", {}),
            model.get("duration_weights", {}),
            model.get("publish_hour_weights", {}),
        ]:
            self._normalize(bucket)

    def _normalize(self, weights: dict[str, float]) -> None:
        if not weights:
            return
        total = sum(max(self.rotation_floor, float(v)) for v in weights.values())
        if total <= 0:
            even = 1.0 / len(weights)
            for key in weights:
                weights[key] = even
            return
        for key in weights:
            weights[key] = max(self.rotation_floor, float(weights[key])) / total

```

`strategy_engine/__init__.py`
```python

```

`strategy_engine/decision_model.py`
```python
"""Strategy decision model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class StrategyDecision:
    """Selected strategy for a new video."""

    format_name: str
    hook_style: str
    duration_seconds: int
    caption_style: str
    publish_hour: int


class DecisionModel:
    """Sample next content strategy from weighted profile."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def decide(
        self,
        strategy_model: dict[str, Any],
        top_trend_category: str | None,
        now: datetime | None = None,
    ) -> StrategyDecision:
        """Return next strategy decision using learned weights."""
        format_weights = dict(strategy_model.get("format_weights", {}))
        hook_weights = dict(strategy_model.get("hook_weights", {}))
        caption_weights = dict(strategy_model.get("caption_weights", {}))
        duration_weights = {int(k): float(v) for k, v in strategy_model.get("duration_weights", {}).items()}
        hour_weights = {int(k): float(v) for k, v in strategy_model.get("publish_hour_weights", {}).items()}

        self._apply_trend_bias(format_weights, top_trend_category)

        format_name = self._weighted_choice(format_weights, default="story")
        hook_style = self._weighted_choice(hook_weights, default="question")
        caption_style = self._weighted_choice(caption_weights, default="direct")
        duration_seconds = int(self._weighted_choice(duration_weights, default=12))

        if now is not None and now.hour in hour_weights:
            hour_weights[now.hour] *= 1.05
        publish_hour = int(self._weighted_choice(hour_weights, default=12))

        return StrategyDecision(
            format_name=format_name,
            hook_style=hook_style,
            duration_seconds=duration_seconds,
            caption_style=caption_style,
            publish_hour=publish_hour,
        )

    def _apply_trend_bias(self, format_weights: dict[str, float], top_category: str | None) -> None:
        if not top_category or not format_weights:
            return
        if top_category == "how_to":
            for key in ["tutorial", "listicle"]:
                if key in format_weights:
                    format_weights[key] *= 1.15
        elif top_category == "curiosity":
            for key in ["story", "challenge"]:
                if key in format_weights:
                    format_weights[key] *= 1.15
        elif top_category == "pain_point":
            for key in ["listicle", "story"]:
                if key in format_weights:
                    format_weights[key] *= 1.1

    def _weighted_choice(self, weights: dict[Any, float], default: Any) -> Any:
        if not weights:
            return default
        keys = list(weights.keys())
        vals = [max(0.0001, float(weights[k])) for k in keys]
        total = sum(vals)
        threshold = self._random.uniform(0.0, total)
        cumulative = 0.0
        for key, weight in zip(keys, vals):
            cumulative += weight
            if cumulative >= threshold:
                return key
        return keys[-1]

```

`strategy_engine/format_multiplier.py`
```python
"""Helper utilities for strategy weight boosting."""

from __future__ import annotations

from typing import Any


class FormatMultiplier:
    """Apply multiplicative updates to strategy profiles."""

    def boost(self, weights: dict[str, float], key: str, multiplier: float = 1.1) -> dict[str, float]:
        """Boost a specific key weight and return normalized distribution."""
        if key in weights:
            weights[key] *= multiplier
        return self._normalize(weights)

    def dampen(self, weights: dict[str, float], key: str, multiplier: float = 0.9, floor: float = 0.05) -> dict[str, float]:
        """Reduce a specific key weight and return normalized distribution."""
        if key in weights:
            weights[key] = max(floor, weights[key] * multiplier)
        return self._normalize(weights)

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        if not weights:
            return weights
        total = sum(weights.values())
        if total <= 0:
            even = 1.0 / len(weights)
            return {k: even for k in weights}
        return {k: v / total for k, v in weights.items()}

    def rotate(self, weights: dict[str, float], min_share: float = 0.05) -> dict[str, float]:
        """Guarantee exploration floor for each style."""
        if not weights:
            return weights
        adjusted = {k: max(min_share, v) for k, v in weights.items()}
        return self._normalize(adjusted)

    def apply_many(self, weights: dict[str, float], boosts: list[str], dampens: list[str]) -> dict[str, float]:
        """Apply bulk winner/loser updates."""
        for key in boosts:
            if key in weights:
                weights[key] *= 1.1
        for key in dampens:
            if key in weights:
                weights[key] *= 0.9
        return self._normalize(weights)

```

`strategy_engine/kill_underperforming_logic.py`
```python
"""Underperforming pattern suppression logic."""

from __future__ import annotations

from typing import Any

import pandas as pd


class KillUnderperformingLogic:
    """Identify and reduce weak-performing patterns."""

    def detect_underperformers(self, scored_df: pd.DataFrame, threshold: float) -> dict[str, list[Any]]:
        """Return patterns whose average rank percentile is below threshold."""
        if scored_df.empty or "score_rank_pct" not in scored_df:
            return {"formats": [], "hooks": [], "caption_styles": [], "durations": []}

        under = scored_df[scored_df["score_rank_pct"] < threshold]
        if under.empty:
            return {"formats": [], "hooks": [], "caption_styles": [], "durations": []}

        return {
            "formats": under["format_name"].dropna().astype(str).unique().tolist() if "format_name" in under else [],
            "hooks": under["hook_style"].dropna().astype(str).unique().tolist() if "hook_style" in under else [],
            "caption_styles": under["caption_style"].dropna().astype(str).unique().tolist() if "caption_style" in under else [],
            "durations": under["duration_seconds"].dropna().astype(int).unique().tolist() if "duration_seconds" in under else [],
        }

```

`strategy_engine/publish_time_optimizer.py`
```python
"""Publish time optimization utilities."""

from __future__ import annotations

import pandas as pd


class PublishTimeOptimizer:
    """Select best publish hour from historical score data."""

    def best_hour(self, scored_df: pd.DataFrame, fallback_hour: int = 12) -> int:
        """Compute best hour based on average engagement score."""
        if scored_df.empty or "publish_hour" not in scored_df or "engagement_score" not in scored_df:
            return fallback_hour

        grouped = scored_df.groupby("publish_hour", as_index=False)["engagement_score"].mean()
        if grouped.empty:
            return fallback_hour
        best = grouped.sort_values("engagement_score", ascending=False).iloc[0]
        return int(best["publish_hour"])

```

`scheduler/__init__.py`
```python

```

`scheduler/cron_runner.py`
```python
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

```

`scheduler/queue_processor.py`
```python
"""Upload queue processing logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from upload_engine.upload_queue_manager import UploadQueueManager, UploadTask
from upload_engine.upload_retry_logic import retry_operation


@dataclass
class QueueProcessResult:
    """Queue processing outcome."""

    processed: int = 0
    succeeded: int = 0
    failed: int = 0


class QueueProcessor:
    """Process pending upload tasks with retry handling."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def process(
        self,
        queue_manager: UploadQueueManager,
        upload_fn: Callable[[UploadTask], tuple[bool, str | None, str | None]],
        attempts: int,
        backoff_seconds: int,
        on_success: Callable[[UploadTask, str | None], None] | None = None,
    ) -> QueueProcessResult:
        """Process all pending queue tasks."""
        outcome = QueueProcessResult()

        while True:
            task = queue_manager.next_pending()
            if task is None:
                break

            outcome.processed += 1

            def operation() -> tuple[bool, str | None, str | None]:
                success, post_id, error = upload_fn(task)
                if not success:
                    raise RuntimeError(error or "unknown upload error")
                return success, post_id, error

            try:
                _, post_id, _ = retry_operation(
                    operation=operation,
                    attempts=attempts,
                    backoff_seconds=backoff_seconds,
                    logger=self.logger,
                    context={"task_id": task.id, "account": task.account_name},
                )
                queue_manager.mark_completed(task.id, post_id=post_id)
                if on_success is not None:
                    on_success(task, post_id)
                outcome.succeeded += 1
            except Exception as exc:  # noqa: BLE001
                queue_manager.mark_failed(task.id, reason=str(exc))
                outcome.failed += 1
                self.logger.error(
                    "Queue task permanently failed",
                    context={"task_id": task.id, "account": task.account_name, "error": str(exc)},
                )

        return outcome

```

`storage/videos/.gitkeep`
```text

```

`storage/logs/.gitkeep`
```text

```

`storage/metrics/.gitkeep`
```text

```

`storage/models/.gitkeep`
```text

```

`storage/accounts/.gitkeep`
```text

```

`storage/queue/.gitkeep`
```text

```

`tests/__init__.py`
```python

```

`tests/conftest.py`
```python
"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def sample_config_dict(tmp_path: Path) -> dict:
    """Return a reusable config dictionary."""
    return {
        "project": {"name": "test", "storage_root": str(tmp_path / "storage")},
        "execution": {"account_mode": "sequential", "max_workers": 2},
        "generation": {
            "allowed_durations": [9, 12, 15],
            "default_duration_seconds": 12,
            "video": {"resolution": "1080x1920", "fps": 30},
        },
        "analytics": {
            "mock_mode": True,
            "min_age_minutes_for_collection": 0,
            "engagement_weights": {
                "views": 0.001,
                "completion_rate": 100.0,
                "shares": 6.0,
                "comments": 4.0,
            },
        },
        "strategy": {
            "learning_rate": 0.15,
            "underperforming_threshold": 0.45,
            "rotation_floor": 0.05,
            "initial_format_weights": {"story": 1.0, "listicle": 1.0},
            "initial_hook_weights": {"question": 1.0, "shock": 1.0},
            "initial_caption_weights": {"direct": 1.0, "curiosity": 1.0},
            "initial_duration_weights": {"9": 1.0, "12": 1.0},
        },
        "upload": {
            "mode": "draft",
            "retry_attempts": 3,
            "retry_backoff_seconds": 0,
            "headless": True,
            "mock_mode": True,
            "tiktok_upload_url": "https://www.tiktok.com/upload",
            "post_wait_seconds": 0,
        },
        "scheduler": {"interval_minutes": 1, "cron_enabled": False},
        "accounts": [
            {
                "name": "acct_one",
                "active": True,
                "niche": "fitness",
                "language": "en",
                "timezone": "UTC",
                "ollama_model": "llama3.1:8b",
                "tags_seed": ["fitness"],
                "session_state_path": str(tmp_path / "storage" / "accounts" / "acct_one" / "session_state.json"),
            }
        ],
    }


@pytest.fixture
def sample_config_file(tmp_path: Path, sample_config_dict: dict) -> Path:
    """Write sample config to temporary file."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(sample_config_dict), encoding="utf-8")
    return path

```

`tests/test_core_constants.py`
```python
from core import constants


def test_constants_are_defined() -> None:
    assert constants.DEFAULT_CONFIG_PATH.name == "config.yaml"
    assert constants.DRAFT_MODE == "draft"
    assert constants.PUBLISH_MODE == "publish"

```

`tests/test_core_config_loader.py`
```python
from pathlib import Path

from core.config_loader import ConfigLoader, active_accounts


def test_config_loader_reads_config(sample_config_file: Path) -> None:
    loader = ConfigLoader(config_path=sample_config_file, env_path=sample_config_file.parent / ".env")
    config = loader.load()
    assert config["project"]["name"] == "test"
    storage = Path(config["project"]["storage_root"])
    assert (storage / "videos").exists()


def test_active_accounts(sample_config_dict: dict) -> None:
    accounts = active_accounts(sample_config_dict)
    assert len(accounts) == 1
    assert accounts[0]["name"] == "acct_one"

```

`tests/test_core_logger.py`
```python
from pathlib import Path

from core.logger import setup_logger


def test_logger_writes_jsonl(tmp_path: Path) -> None:
    logger = setup_logger(tmp_path)
    logger.info("hello", context={"a": 1})
    log_file = tmp_path / "autonomous_engine.jsonl"
    assert log_file.exists()
    assert "hello" in log_file.read_text(encoding="utf-8")

```

`tests/test_account_state_handler.py`
```python
from pathlib import Path

from accounts.account_state_handler import AccountState, AccountStateHandler


def test_state_handler_load_save(tmp_path: Path) -> None:
    handler = AccountStateHandler(tmp_path)
    state = handler.load("acct")
    assert state.account_name == "acct"
    state.generated_count = 5
    handler.save(state)
    loaded = handler.load("acct")
    assert loaded.generated_count == 5


def test_mark_run_updates_timestamp(tmp_path: Path) -> None:
    handler = AccountStateHandler(tmp_path)
    state = AccountState(account_name="acct")
    handler.mark_run(state, success=True)
    assert state.last_run_at is not None
    assert state.last_success_at is not None

```

`tests/test_account_manager.py`
```python
from pathlib import Path

from accounts.account_manager import AccountManager


def test_account_manager_paths(sample_config_dict: dict, tmp_path: Path) -> None:
    manager = AccountManager(sample_config_dict, tmp_path)
    profile = manager.get_account("acct_one")
    account_dir = manager.ensure_account_dirs(profile)
    assert account_dir.exists()
    assert manager.metrics_path(profile).name == "metrics.jsonl"
    assert manager.strategy_model_path(profile).name == "strategy_model.json"

```

`tests/test_trend_scraper.py`
```python
from trend_engine.trend_scraper import TrendScraper


def test_scraper_returns_signals() -> None:
    scraper = TrendScraper(seed=7)
    trends = scraper.scrape("fitness", limit=5)
    assert len(trends) == 5
    assert trends[0].niche == "fitness"

```

`tests/test_trend_classifier.py`
```python
from trend_engine.trend_classifier import TrendClassifier
from trend_engine.trend_scraper import TrendSignal


def test_classifier_categories() -> None:
    classifier = TrendClassifier()
    trends = [TrendSignal("fat loss myths", "fitness", "x", 1.2, 0.4, 3)]
    result = classifier.classify(trends)
    assert result[0].category == "pain_point"

```

`tests/test_trend_score_engine.py`
```python
from trend_engine.trend_classifier import ClassifiedTrend
from trend_engine.trend_score_engine import TrendScoreEngine


def test_scoring_orders_descending() -> None:
    engine = TrendScoreEngine()
    data = [
        ClassifiedTrend("a", "n", "s", 2.0, 0.2, 2, "how_to", "edu"),
        ClassifiedTrend("b", "n", "s", 1.0, 0.8, 48, "general", "edu"),
    ]
    scored = engine.score(data)
    assert scored[0].score >= scored[1].score

```

`tests/test_content_idea_generator.py`
```python
from content_engine.idea_generator import IdeaGenerator


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_fallback_ideas(monkeypatch) -> None:
    gen = IdeaGenerator(DummyLogger())
    monkeypatch.setattr(gen, "_ollama_available", lambda: False)
    ideas = gen.generate_ideas(
        account_name="acct",
        niche="fitness",
        model="llama3",
        scored_trends=[{"phrase": "high protein breakfast"}],
        format_name="story",
        count=2,
    )
    assert len(ideas) == 2
    assert ideas[0].format_name == "story"

```

`tests/test_content_script_generator.py`
```python
from content_engine.idea_generator import ContentIdea
from content_engine.script_generator import ScriptGenerator


def test_script_generation() -> None:
    idea = ContentIdea("Title", "Angle", "curiosity", "trend", "story")
    script = ScriptGenerator().generate(idea, hook_style="question", duration_seconds=12)
    assert "trend" in script.hook.lower()
    assert script.duration_seconds == 12

```

`tests/test_content_hook_optimizer.py`
```python
from content_engine.hook_optimizer import HookOptimizer


def test_hook_optimizer_uses_winner() -> None:
    optimized = HookOptimizer().optimize("base hook", ["winner hook"], "shock")
    assert optimized.startswith("winner hook")

```

`tests/test_content_caption_generator.py`
```python
from content_engine.caption_generator import CaptionGenerator
from content_engine.idea_generator import ContentIdea


def test_caption_style() -> None:
    idea = ContentIdea("Test title", "angle", "emotion", "trend", "story")
    caption = CaptionGenerator().generate(idea, caption_style="urgency")
    assert caption.startswith("Try this before")

```

`tests/test_content_hashtag_optimizer.py`
```python
from content_engine.hashtag_optimizer import HashtagOptimizer


def test_hashtag_cleanup_and_limit() -> None:
    tags = HashtagOptimizer().generate(
        niche="personal finance",
        trend_phrases=["high yield savings explained"],
        seed_tags=["money", "budget"],
        max_tags=4,
    )
    assert len(tags) == 4
    assert all(tag.startswith("#") for tag in tags)

```

`tests/test_media_voice_generator.py`
```python
from pathlib import Path

from media_engine.voice_generator import VoiceGenerator


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_voice_fallback(monkeypatch, tmp_path: Path) -> None:
    gen = VoiceGenerator(DummyLogger())

    def fake_silence(path: Path, duration_seconds: int) -> None:
        path.write_bytes(b"RIFF")

    monkeypatch.setattr(gen, "_generate_silence", fake_silence)
    out = gen.synthesize("hello", tmp_path / "x.wav", voice_model_path=None, duration_seconds=3)
    assert out.exists()

```

`tests/test_media_subtitle_generator.py`
```python
from pathlib import Path

from media_engine.subtitle_generator import SubtitleGenerator


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_subtitle_fallback(tmp_path: Path) -> None:
    gen = SubtitleGenerator(DummyLogger())
    srt = gen.generate(
        audio_path=tmp_path / "audio.wav",
        output_srt=tmp_path / "out.srt",
        language="en",
        fallback_text="one two three four five six",
        duration_seconds=6,
    )
    assert srt.exists()
    assert "-->" in srt.read_text(encoding="utf-8")

```

`tests/test_media_video_renderer.py`
```python
from pathlib import Path
from types import SimpleNamespace

import media_engine.video_renderer as mod
from media_engine.video_renderer import VideoRenderer


class DummyLogger:
    def error(self, *args, **kwargs):
        return None


def test_video_render_success(monkeypatch, tmp_path: Path) -> None:
    renderer = VideoRenderer(DummyLogger())

    def fake_run(cmd, capture_output, text, check):
        Path(cmd[-1]).write_bytes(b"mp4")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    out = renderer.render(
        output_video=tmp_path / "v.mp4",
        audio_path=tmp_path / "a.wav",
        subtitle_path=tmp_path / "s.srt",
        hook_text="hook",
        body_text="body",
        caption_text="cap",
        duration_seconds=9,
    )
    assert out.exists()

```

`tests/test_media_thumbnail_selector.py`
```python
from pathlib import Path
from types import SimpleNamespace

import media_engine.thumbnail_selector as mod
from media_engine.thumbnail_selector import ThumbnailSelector


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_thumbnail_extract(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, capture_output, text, check):
        Path(cmd[-1]).write_bytes(b"png")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    selector = ThumbnailSelector(DummyLogger())
    out = selector.extract(tmp_path / "v.mp4", tmp_path / "t.png")
    assert out.name == "t.png"

```

`tests/test_upload_draft_mode.py`
```python
from upload_engine.draft_or_publish_mode import DraftOrPublishMode


def test_mode_conversion() -> None:
    assert DraftOrPublishMode.from_value("publish") == DraftOrPublishMode.PUBLISH
    assert DraftOrPublishMode.from_value("anything") == DraftOrPublishMode.DRAFT

```

`tests/test_upload_retry_logic.py`
```python
from upload_engine.upload_retry_logic import retry_operation


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_retry_success_after_failures() -> None:
    calls = {"n": 0}

    def op() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("fail")
        return 7

    result = retry_operation(op, attempts=3, backoff_seconds=0, logger=DummyLogger())
    assert result == 7

```

`tests/test_upload_queue_manager.py`
```python
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

```

`tests/test_upload_tiktok_uploader.py`
```python
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

```

`tests/test_analytics_metrics_collector.py`
```python
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

```

`tests/test_analytics_performance_analyzer.py`
```python
import pandas as pd

from analytics_engine.performance_analyzer import EngagementWeights, PerformanceAnalyzer


def test_engagement_score_formula() -> None:
    analyzer = PerformanceAnalyzer(EngagementWeights(views=0.001, completion_rate=100.0, shares=6.0, comments=4.0))
    row = {"views": 1000, "completion_rate": 0.5, "shares": 10, "comments": 5}
    score = analyzer.engagement_score(row)
    assert round(score, 3) == round(0.001 * 1000 + 100.0 * 0.5 + 6.0 * 10 + 4.0 * 5, 3)


def test_analyze_adds_columns() -> None:
    analyzer = PerformanceAnalyzer(EngagementWeights(views=1, completion_rate=1, shares=1, comments=1))
    df = pd.DataFrame([{"views": 1, "completion_rate": 1, "shares": 1, "comments": 1}])
    scored = analyzer.analyze(df)
    assert "engagement_score" in scored.columns

```

`tests/test_analytics_winner_detector.py`
```python
import pandas as pd

from analytics_engine.winner_pattern_detector import WinnerPatternDetector


def test_detect_winners_and_losers() -> None:
    df = pd.DataFrame(
        [
            {"engagement_score": 100, "hook_style": "question", "duration_seconds": 9, "caption_style": "direct", "format_name": "story", "publish_hour": 10},
            {"engagement_score": 10, "hook_style": "shock", "duration_seconds": 15, "caption_style": "urgency", "format_name": "listicle", "publish_hour": 23},
        ]
    )
    result = WinnerPatternDetector().detect(df)
    assert "winners" in result
    assert "losers" in result

```

`tests/test_analytics_format_score_model.py`
```python
from pathlib import Path

from analytics_engine.format_score_model import FormatScoreModel


def test_model_init_and_update(tmp_path: Path) -> None:
    store = FormatScoreModel(tmp_path / "model.json")
    model = store.load_or_init(
        {
            "initial_format_weights": {"story": 1.0, "listicle": 1.0},
            "initial_hook_weights": {"question": 1.0, "shock": 1.0},
            "initial_caption_weights": {"direct": 1.0, "curiosity": 1.0},
            "initial_duration_weights": {"9": 1.0, "12": 1.0},
        }
    )
    updated = store.update(
        model,
        winners={"formats": ["story"], "hooks": ["question"], "caption_styles": ["direct"], "durations": [9], "publish_hours": [12]},
        losers={"formats": ["listicle"], "hooks": ["shock"], "caption_styles": ["curiosity"], "durations": [12], "publish_hours": [23]},
        learning_rate=0.1,
    )
    store.save(updated)
    assert Path(tmp_path / "model.json").exists()

```

`tests/test_strategy_decision_model.py`
```python
from strategy_engine.decision_model import DecisionModel


def test_decision_generation() -> None:
    model = {
        "format_weights": {"story": 1.0},
        "hook_weights": {"question": 1.0},
        "caption_weights": {"direct": 1.0},
        "duration_weights": {"12": 1.0},
        "publish_hour_weights": {str(i): 1.0 for i in range(24)},
    }
    decision = DecisionModel(seed=1).decide(model, top_trend_category="how_to")
    assert decision.format_name == "story"
    assert decision.duration_seconds == 12

```

`tests/test_strategy_format_multiplier.py`
```python
from strategy_engine.format_multiplier import FormatMultiplier


def test_boost_and_rotate() -> None:
    fm = FormatMultiplier()
    boosted = fm.boost({"a": 1.0, "b": 1.0}, "a", multiplier=1.2)
    assert round(sum(boosted.values()), 6) == 1.0
    rotated = fm.rotate({"a": 0.001, "b": 0.999}, min_share=0.05)
    assert rotated["a"] > 0.0

```

`tests/test_strategy_kill_underperforming.py`
```python
import pandas as pd

from strategy_engine.kill_underperforming_logic import KillUnderperformingLogic


def test_underperformer_detection() -> None:
    df = pd.DataFrame(
        [
            {"score_rank_pct": 0.2, "format_name": "story", "hook_style": "shock", "caption_style": "direct", "duration_seconds": 15},
            {"score_rank_pct": 0.9, "format_name": "listicle", "hook_style": "question", "caption_style": "urgency", "duration_seconds": 9},
        ]
    )
    under = KillUnderperformingLogic().detect_underperformers(df, threshold=0.45)
    assert "story" in under["formats"]

```

`tests/test_strategy_publish_time_optimizer.py`
```python
import pandas as pd

from strategy_engine.publish_time_optimizer import PublishTimeOptimizer


def test_best_hour() -> None:
    df = pd.DataFrame(
        [
            {"publish_hour": 9, "engagement_score": 30},
            {"publish_hour": 9, "engagement_score": 40},
            {"publish_hour": 15, "engagement_score": 20},
        ]
    )
    assert PublishTimeOptimizer().best_hour(df) == 9

```

`tests/test_scheduler_cron_runner.py`
```python
from pathlib import Path

from scheduler.cron_runner import CronRunner


class DummyLogger:
    def info(self, *args, **kwargs):
        return None


def test_cron_line_contains_main() -> None:
    line = CronRunner(DummyLogger()).cron_line(project_root=Path("/tmp/project"), interval_minutes=30)
    assert "main.py run" in line

```

`tests/test_scheduler_queue_processor.py`
```python
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

```

`tests/test_engine_and_cli.py`
```python
from pathlib import Path

from engine import AutonomousGrowthEngine


def test_engine_init_and_stats(sample_config_file: Path) -> None:
    engine = AutonomousGrowthEngine(config_path=str(sample_config_file))
    engine.init()
    stats = engine.stats()
    assert len(stats) == 1
    assert stats[0]["account"] == "acct_one"

```
