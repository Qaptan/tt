from content_engine.hook_optimizer import HookOptimizer


def test_hook_optimizer_uses_winner() -> None:
    optimized = HookOptimizer().optimize("base hook", ["winner hook"], "shock")
    assert optimized.startswith("winner hook")
