from core import constants


def test_constants_are_defined() -> None:
    assert constants.DEFAULT_CONFIG_PATH.name == "config.yaml"
    assert constants.DRAFT_MODE == "draft"
    assert constants.PUBLISH_MODE == "publish"
