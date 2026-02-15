from pathlib import Path

from core.logger import setup_logger


def test_logger_writes_jsonl(tmp_path: Path) -> None:
    logger = setup_logger(tmp_path)
    logger.info("hello", context={"a": 1})
    log_file = tmp_path / "autonomous_engine.jsonl"
    assert log_file.exists()
    assert "hello" in log_file.read_text(encoding="utf-8")
