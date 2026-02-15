from content_engine.hashtag_optimizer import HashtagOptimizer


def test_hashtag_cleanup_and_limit() -> None:
    tags = HashtagOptimizer().generate(
        niche="personal finance",
        trend_phrases=["high yield savings explained"],
        seed_tags=["money", "budget"],
        max_tags=4,
    )
    assert len(tags) == 4
    assert all(tag.startswith("#") for tag in tags)
