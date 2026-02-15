from strategy_engine.format_multiplier import FormatMultiplier


def test_boost_and_rotate() -> None:
    fm = FormatMultiplier()
    boosted = fm.boost({"a": 1.0, "b": 1.0}, "a", multiplier=1.2)
    assert round(sum(boosted.values()), 6) == 1.0
    rotated = fm.rotate({"a": 0.001, "b": 0.999}, min_share=0.05)
    assert rotated["a"] > 0.0
