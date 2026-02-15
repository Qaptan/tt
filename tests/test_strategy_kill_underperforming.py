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
