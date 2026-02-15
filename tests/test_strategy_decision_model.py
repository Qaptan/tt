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
