from pathlib import Path

from accounts.account_manager import AccountManager


def test_account_manager_paths(sample_config_dict: dict, tmp_path: Path) -> None:
    manager = AccountManager(sample_config_dict, tmp_path)
    profile = manager.get_account("acct_one")
    account_dir = manager.ensure_account_dirs(profile)
    assert account_dir.exists()
    assert manager.metrics_path(profile).name == "metrics.jsonl"
    assert manager.strategy_model_path(profile).name == "strategy_model.json"
