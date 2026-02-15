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
