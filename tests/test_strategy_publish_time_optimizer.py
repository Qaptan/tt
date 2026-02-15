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
