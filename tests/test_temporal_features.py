import unittest
from datetime import datetime, timedelta

from features import HistoricalStatsCache, build_match_features


class HistoricalStatsCacheTest(unittest.TestCase):
    def setUp(self):
        self.base = datetime(2026, 1, 1, 12, 0, 0)
        self.cache = HistoricalStatsCache(
            {
                (1, 100, True): [
                    {"footystats_id": 1, "match_date": self.base, "hsh_result": "H1", "hsh_goals_h1": 1, "hsh_goals_h2": 0},
                    {"footystats_id": 2, "match_date": self.base + timedelta(days=7), "hsh_result": "EQ", "hsh_goals_h1": 0, "hsh_goals_h2": 0},
                    {"footystats_id": 3, "match_date": self.base + timedelta(days=14), "hsh_result": "H2", "hsh_goals_h1": 0, "hsh_goals_h2": 1},
                ],
                (2, 100, False): [
                    {"footystats_id": 4, "match_date": self.base, "hsh_result": "H2", "hsh_goals_h1": 0, "hsh_goals_h2": 1},
                    {"footystats_id": 5, "match_date": self.base + timedelta(days=7), "hsh_result": "EQ", "hsh_goals_h1": 1, "hsh_goals_h2": 1},
                    {"footystats_id": 6, "match_date": self.base + timedelta(days=14), "hsh_result": "H1", "hsh_goals_h1": 1, "hsh_goals_h2": 0},
                ],
            }
        )

    def test_build_match_features_uses_only_prior_matches(self):
        match = {
            "home_id": 1,
            "away_id": 2,
            "league_id": 100,
            "footystats_id": 999,
            "match_date": self.base + timedelta(days=14),
            "home_ppg": 1.8,
            "away_ppg": 1.2,
        }

        features = build_match_features(match, "A", stats_cache=self.cache)

        self.assertAlmostEqual(features["home_pct_h1"], 0.5)
        self.assertAlmostEqual(features["home_pct_eq"], 0.5)
        self.assertAlmostEqual(features["home_pct_h2"], 0.0)
        self.assertAlmostEqual(features["away_pct_h2"], 0.5)
        self.assertAlmostEqual(features["away_pct_eq"], 0.5)
        self.assertAlmostEqual(features["ppg_diff"], 0.6)

    def test_compute_team_hsh_stats_respects_last_n(self):
        stats = self.cache.compute_team_hsh_stats(
            team_id=1,
            league_id=100,
            as_home=True,
            last_n=1,
            before_date=self.base + timedelta(days=21),
        )

        self.assertAlmostEqual(stats["home_pct_h2"], 1.0)
        self.assertAlmostEqual(stats["home_avg_goals_h2"], 1.0)
        self.assertEqual(stats["home_n_matches"], 1)


if __name__ == "__main__":
    unittest.main()
