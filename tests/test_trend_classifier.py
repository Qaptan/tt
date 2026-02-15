from trend_engine.trend_classifier import TrendClassifier
from trend_engine.trend_scraper import TrendSignal


def test_classifier_categories() -> None:
    classifier = TrendClassifier()
    trends = [TrendSignal("fat loss myths", "fitness", "x", 1.2, 0.4, 3)]
    result = classifier.classify(trends)
    assert result[0].category == "pain_point"
