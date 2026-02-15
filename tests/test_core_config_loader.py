from pathlib import Path

from core.config_loader import ConfigLoader, active_accounts


def test_config_loader_reads_config(sample_config_file: Path) -> None:
    loader = ConfigLoader(config_path=sample_config_file, env_path=sample_config_file.parent / ".env")
    config = loader.load()
    assert config["project"]["name"] == "test"
    storage = Path(config["project"]["storage_root"])
    assert (storage / "videos").exists()


def test_active_accounts(sample_config_dict: dict) -> None:
    accounts = active_accounts(sample_config_dict)
    assert len(accounts) == 1
    assert accounts[0]["name"] == "acct_one"
