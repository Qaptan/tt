"""Hook optimization module."""

from __future__ import annotations


class HookOptimizer:
    """Optimize hooks using historical winner patterns."""

    def optimize(self, base_hook: str, winning_hooks: list[str], hook_style: str) -> str:
        """Return optimized hook with lightweight adaptation."""
        if winning_hooks:
            best = winning_hooks[0]
            return f"{best} {base_hook}"
        if hook_style == "shock":
            return f"Wait. {base_hook}"
        if hook_style == "question":
            return f"Quick question: {base_hook}"
        return base_hook
