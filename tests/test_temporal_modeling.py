import unittest
from datetime import datetime, timedelta

import pandas as pd

from model import (
    _iter_walk_forward_windows,
    _segment_matches_by_season,
    _select_latest_usable_segment,
    split_temporal_dataset,
)


class TemporalModelingTest(unittest.TestCase):
    def test_split_temporal_dataset_is_chronological(self):
        base = datetime(2026, 1, 1, 12, 0, 0)
        df = pd.DataFrame(
            {
                "match_date": [base + timedelta(days=day) for day in range(120)],
                "label": ["H1"] * 120,
            }
        )

        train_df, valid_df, test_df = split_temporal_dataset(
            df,
            train_days=60,
            valid_days=14,
            test_days=7,
        )

        self.assertEqual(len(train_df), 60)
        self.assertEqual(len(valid_df), 14)
        self.assertEqual(len(test_df), 8)
        self.assertLess(train_df["match_date"].max(), valid_df["match_date"].min())
        self.assertLess(valid_df["match_date"].max(), test_df["match_date"].min())

    def test_split_temporal_dataset_supports_match_counts(self):
        base = datetime(2026, 1, 1, 12, 0, 0)
        df = pd.DataFrame(
            {
                "match_date": [base + timedelta(days=day) for day in range(40)],
                "label": ["H1"] * 40,
            }
        )

        train_df, valid_df, test_df = split_temporal_dataset(
            df,
            protocol_mode="matches",
            train_matches=20,
            valid_matches=8,
            test_matches=5,
        )

        self.assertEqual(len(train_df), 20)
        self.assertEqual(len(valid_df), 8)
        self.assertEqual(len(test_df), 5)
        self.assertLess(train_df["match_date"].max(), valid_df["match_date"].min())
        self.assertLess(valid_df["match_date"].max(), test_df["match_date"].min())

    def test_select_latest_usable_segment_skips_sparse_latest_tail(self):
        base = datetime(2026, 1, 1, 12, 0, 0)
        segment_old = [
            {"footystats_id": idx, "match_date": base + timedelta(days=idx)}
            for idx in range(5)
        ]
        segment_usable = [
            {"footystats_id": 100 + idx, "match_date": base + timedelta(days=60 + idx * 5)}
            for idx in range(10)
        ]
        segment_sparse = [
            {"footystats_id": 999, "match_date": base + timedelta(days=200)}
        ]
        matches = segment_old + segment_usable + segment_sparse

        selected, meta = _select_latest_usable_segment(
            matches,
            valid_days=14,
            test_days=7,
            step_days=7,
            max_gap_days=21,
            segment_mode="gap",
        )

        self.assertEqual(meta["segment_count"], 3)
        self.assertEqual(meta["segment_index"], 1)
        self.assertEqual(len(selected), len(segment_usable))
        self.assertEqual(selected[0]["footystats_id"], segment_usable[0]["footystats_id"])
        self.assertEqual(selected[-1]["footystats_id"], segment_usable[-1]["footystats_id"])

    def test_segment_matches_by_season_keeps_winter_break_inside_same_segment(self):
        base = datetime(2016, 8, 13, 12, 30, 0)
        matches = [
            {"footystats_id": 1, "match_date": base, "season": "20162017"},
            {"footystats_id": 2, "match_date": base + timedelta(days=120), "season": "20162017"},
            {"footystats_id": 3, "match_date": base + timedelta(days=240), "season": "20172018"},
        ]

        segments = _segment_matches_by_season(matches)

        self.assertEqual(len(segments), 2)
        self.assertEqual([row["footystats_id"] for row in segments[0]], [1, 2])
        self.assertEqual([row["footystats_id"] for row in segments[1]], [3])

    def test_iter_walk_forward_windows_supports_match_counts(self):
        base = datetime(2026, 1, 1, 12, 0, 0)
        df = pd.DataFrame(
            {
                "match_date": [base + timedelta(days=day) for day in range(36)],
                "label": ["H1"] * 36,
            }
        )

        folds = list(
            _iter_walk_forward_windows(
                df,
                protocol_mode="matches",
                train_matches=12,
                valid_matches=6,
                test_matches=4,
                step_matches=4,
            )
        )

        self.assertEqual(len(folds), 5)
        _, train_df, valid_df, test_df = folds[0]
        self.assertEqual(len(train_df), 12)
        self.assertEqual(len(valid_df), 6)
        self.assertEqual(len(test_df), 4)


if __name__ == "__main__":
    unittest.main()
