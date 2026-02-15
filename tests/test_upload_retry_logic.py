from upload_engine.upload_retry_logic import retry_operation


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None


def test_retry_success_after_failures() -> None:
    calls = {"n": 0}

    def op() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("fail")
        return 7

    result = retry_operation(op, attempts=3, backoff_seconds=0, logger=DummyLogger())
    assert result == 7
