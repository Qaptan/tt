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
