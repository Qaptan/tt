from pathlib import Path

from accounts.account_state_handler import AccountState, AccountStateHandler


def test_state_handler_load_save(tmp_path: Path) -> None:
    handler = AccountStateHandler(tmp_path)
    state = handler.load("acct")
    assert state.account_name == "acct"
    state.generated_count = 5
    handler.save(state)
    loaded = handler.load("acct")
    assert loaded.generated_count == 5


def test_mark_run_updates_timestamp(tmp_path: Path) -> None:
    handler = AccountStateHandler(tmp_path)
    state = AccountState(account_name="acct")
    handler.mark_run(state, success=True)
    assert state.last_run_at is not None
    assert state.last_success_at is not None
