from strategy_engine.decision_model import DecisionModel


def test_thompson_sampling_prefers_stronger_posterior() -> None:
    model = {
        "format_weights": {"story": 1.0, "tutorial": 1.0},
        "hook_weights": {"question": 1.0},
        "caption_weights": {"direct": 1.0},
        "duration_weights": {"12": 1.0},
        "publish_hour_weights": {str(i): 1.0 for i in range(24)},
        "bandit_state": {
            "format": {
                "story": {"alpha": 2.0, "beta": 8.0},
                "tutorial": {"alpha": 9.0, "beta": 2.0},
            },
            "hook": {"question": {"alpha": 1.0, "beta": 1.0}},
            "caption": {"direct": {"alpha": 1.0, "beta": 1.0}},
            "duration": {"12": {"alpha": 1.0, "beta": 1.0}},
            "publish_hour": {str(i): {"alpha": 1.0, "beta": 1.0} for i in range(24)},
        },
    }

    picks = {"story": 0, "tutorial": 0}
    decider = DecisionModel(seed=7)
    for _ in range(200):
        decision = decider.decide(model, top_trend_category=None)
        picks[decision.format_name] += 1

    assert picks["tutorial"] > picks["story"]
