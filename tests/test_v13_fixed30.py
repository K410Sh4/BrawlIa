"""Perfil de 30 FPS fixos e autoavaliação sem amostras PNG."""

import unittest

from brawlla.config import SessionOptions
from brawlla.self_review import SelfReview


class FixedThirtyFpsTests(unittest.TestCase):
    def test_default_session_rates_are_30(self):
        options = SessionOptions()
        self.assertFalse(options.auto_fps)
        self.assertEqual(options.capture_fps, 30)
        self.assertEqual(options.analysis_fps, 30)
        self.assertEqual(options.video_fps, 30)
        self.assertFalse(options.record_samples)

    def test_review_events_have_no_image_data_or_invented_label(self):
        review = SelfReview()
        review.observe(
            {
                "session_elapsed_s": 2.0,
                "game_state": {"name": "combate"},
                "player_health_bar_candidate": None,
                "match_roster": {"status": "detected", "confirmed_names": 0},
                "timings_ms": {"total": 75.0},
                "tracked_objects": [],
            }
        )
        details = review.snapshot()
        self.assertEqual(details["frames_seen"], 1)
        self.assertEqual(details["combat_frames_without_player"], 1)
        self.assertIn("player_not_detected", details["flag_counts"])
        self.assertIn("roster_without_confirmed_names", details["flag_counts"])
        self.assertFalse(details["saved_png"])
        self.assertFalse(details["weights_updated"])
        self.assertFalse(details["automatic_retraining"])
        self.assertTrue(details["review_points"])

    def test_ambiguous_hp_flagged_and_events_throttled(self):
        review = SelfReview(min_gap_s=2.0)
        for elapsed in (0.5, 1.0, 1.5):
            review.observe(
                {
                    "session_elapsed_s": elapsed,
                    "game_state": {"name": "combate"},
                    "player_health_bar_candidate": {
                        "name_similarity": 0.3,
                        "health_status": "low_fill_candidate",
                    },
                    "match_roster": {},
                    "track_events": [{"event": "lost", "track_id": 7}],
                }
            )
        data = review.snapshot()
        self.assertEqual(data["flag_counts"]["low_hp_unconfirmed_identity"], 3)
        flagged = [
            event
            for event in data["review_points"]
            if event["reason"] == "low_hp_unconfirmed_identity"
        ]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(data["combat_frames_without_player"], 0)

    def test_review_only_keeps_bounded_metadata(self):
        review = SelfReview(max_events=2, min_gap_s=0)
        for index in range(20):
            review.observe(
                {
                    "session_elapsed_s": float(index),
                    "game_state": {"name": "combate"},
                    "player_health_bar_candidate": None,
                }
            )
        self.assertEqual(len(review.snapshot()["review_points"]), 2)
        self.assertEqual(review.snapshot()["flag_counts"]["player_not_detected"], 20)

    def test_review_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            SelfReview(max_events=0)
        with self.assertRaises(ValueError):
            SelfReview(min_gap_s=-1)


if __name__ == "__main__":
    unittest.main()
