from upload_engine.draft_or_publish_mode import DraftOrPublishMode


def test_mode_conversion() -> None:
    assert DraftOrPublishMode.from_value("publish") == DraftOrPublishMode.PUBLISH
    assert DraftOrPublishMode.from_value("anything") == DraftOrPublishMode.DRAFT
