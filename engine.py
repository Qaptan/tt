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
            api_base_url=str(upload_cfg.get("api_base_url", "https://open.tiktokapis.com")),
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
