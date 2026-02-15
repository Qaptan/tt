# tiktok-autonomous-growth-engine

Fully local, modular, multi-account TikTok automation engine for Linux.

## Features
- Local AI pipeline: Ollama + Piper + Whisper + FFmpeg
- Official TikTok Content Posting API upload with OAuth refresh (draft/publish mode)
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

## Installation
```bash
cd tiktok-autonomous-growth-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

## Connect TikTok Session (OAuth Tokens)
For each account, store OAuth token payload in `session_state_path` as JSON:
```bash
cat > storage/accounts/fitness_account/session_state.json <<'EOF'
{
  "access_token": "YOUR_ACCESS_TOKEN",
  "refresh_token": "YOUR_REFRESH_TOKEN",
  "expires_at": "2026-02-15T00:00:00+00:00"
}
EOF
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
python main.py web --host 0.0.0.0 --port 8787
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

## Web Dashboard
Start the visual control panel:
```bash
python main.py web --host 0.0.0.0 --port 8787
```

Open:
```text
http://localhost:8787
```

Dashboard actions:
- `Init` initializes per-account storage and state.
- `Run Cycle` executes generation/upload/learning.
- `Analyze` computes engagement and winner/loser patterns.
- `Retrain` updates strategy weights from collected metrics.
- `Hesap Yonetimi` panelinden hesap ekleme/guncelleme/silme islemleri dogrudan `config.yaml` dosyasina yazilir.

## Troubleshooting
- Upload fails immediately:
  - Check `upload.mock_mode`, `upload.api_base_url`, and `session_state_path`
  - Ensure `.env` has `TIKTOK_CLIENT_KEY` and `TIKTOK_CLIENT_SECRET`
  - Ensure token JSON includes `access_token` and `refresh_token`
- Video rendering fails:
  - Verify `ffmpeg` is on PATH (`which ffmpeg`)
  - Inspect JSON logs in `storage/logs/autonomous_engine.jsonl`
- Ollama fallback always used:
  - Ensure `ollama` binary is installed and model exists (`ollama list`)
- No metrics collected:
  - With `analytics.mock_mode: false`, pending metrics remain pending until real data integration

## Testing
```bash
pytest
```
