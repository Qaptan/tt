import pandas as pd

from analytics_engine.performance_analyzer import EngagementWeights, PerformanceAnalyzer


def test_engagement_score_formula() -> None:
    analyzer = PerformanceAnalyzer(EngagementWeights(views=0.001, completion_rate=100.0, shares=6.0, comments=4.0))
    row = {"views": 1000, "completion_rate": 0.5, "shares": 10, "comments": 5}
    score = analyzer.engagement_score(row)
    assert round(score, 3) == round(0.001 * 1000 + 100.0 * 0.5 + 6.0 * 10 + 4.0 * 5, 3)


def test_analyze_adds_columns() -> None:
    analyzer = PerformanceAnalyzer(EngagementWeights(views=1, completion_rate=1, shares=1, comments=1))
    df = pd.DataFrame([{"views": 1, "completion_rate": 1, "shares": 1, "comments": 1}])
    scored = analyzer.analyze(df)
    assert "engagement_score" in scored.columns
