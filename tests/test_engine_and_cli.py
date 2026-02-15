from pathlib import Path

from engine import AutonomousGrowthEngine


def test_engine_init_and_stats(sample_config_file: Path) -> None:
    engine = AutonomousGrowthEngine(config_path=str(sample_config_file))
    engine.init()
    stats = engine.stats()
    assert len(stats) == 1
    assert stats[0]["account"] == "acct_one"
