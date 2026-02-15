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
