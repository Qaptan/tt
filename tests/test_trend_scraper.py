from trend_engine.trend_scraper import TrendScraper


def test_scraper_returns_signals() -> None:
    scraper = TrendScraper(seed=7)
    trends = scraper.scrape("fitness", limit=5)
    assert len(trends) == 5
    assert trends[0].niche == "fitness"
