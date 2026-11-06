"""Identidade visual associada ao nome, independente da barra e do HP.

Não há OCR nem inferência de identidade pessoal: nomes ambíguos continuam
usando o rastreamento geométrico, sem duplicar observações.
"""

from dataclasses import replace
import unittest

import cv2

from brawlla.config import TrackingConfig, VisionConfig
from brawlla.player import PlayerLocator
from brawlla.tracking import TargetTracker
from test_v09 import sample


KEY = "visual:0123456789abcdef0123"


def player(x=640, y=280, score=0.8, key=KEY, hp="fill_visible"):
    return {
        "kind": "jogador_candidato",
        "center": [x, y],
        "x": x - 35,
        "y": y - 4,
        "width": 70,
        "height": 8,
        "name_identity_key": key,
        "name_similarity": score,
        "name_identity_source": "visual_template",
        "health_status": hp,
    }


def box(x=640, y=280):
    return {
        "kind": "caixa_provavel",
        "center": [x, y],
        "x": x - 35,
        "y": y - 4,
        "width": 70,
        "height": 8,
    }


class PlayerVisualNameTests(unittest.TestCase):
    def test_name_key_stays_fixed_when_health_fill_and_color_change(self):
        locator = PlayerLocator()
        keys = []
        for i, fill in enumerate((70, 35, 8, 2, 35)):
            hsv = cv2.cvtColor(sample(fill=fill), cv2.COLOR_BGR2HSV)
            observation = locator.locate(hsv, timestamp_s=i / 15)
            self.assertIsNotNone(observation)
            self.assertIsNone(observation["name_text"])
            self.assertEqual(observation["name_identity_source"], "visual_template")
            keys.append(observation["name_identity_key"])
        self.assertEqual(len(set(keys)), 1)
        self.assertTrue(keys[0].startswith("visual:"))

    def test_visual_key_only_attached_to_observed_name_and_bar(self):
        locator = PlayerLocator()
        self.assertIsNone(
            locator.locate(
                cv2.cvtColor(sample(name=False), cv2.COLOR_BGR2HSV),
                timestamp_s=0,
            )
        )
        result = locator.locate(
            cv2.cvtColor(sample(), cv2.COLOR_BGR2HSV), timestamp_s=0.1
        )
        self.assertIsNotNone(result.get("name_identity_key"))

    def test_reset_drops_reference_name_and_key(self):
        locator = PlayerLocator(VisionConfig())
        locator.locate(cv2.cvtColor(sample(), cv2.COLOR_BGR2HSV), 0)
        self.assertIsNotNone(locator.name_visual_key)
        locator.reset()
        self.assertIsNone(locator.name_visual_key)


class TrackerNameAnchorTests(unittest.TestCase):
    def test_same_name_keeps_track_when_health_and_position_jump(self):
        tracker = TargetTracker()
        first = tracker.update([player(hp="fill_visible")], timestamp_s=0)
        tid = first[0]["track_id"]
        self.assertEqual(first[0]["identity_association"], "geometria")
        moved = tracker.update(
            [player(x=592, y=355, score=0.3, hp="low_fill_candidate")],
            timestamp_s=0.08,
        )
        # O outlier ainda pode ser associado pela posição, não pelo nome.
        self.assertEqual(moved[0]["track_id"], tid)
        recovered = tracker.update(
            [player(x=641, y=286, score=0.75, hp="fill_visible")],
            timestamp_s=0.16,
        )
        self.assertEqual(recovered[0]["track_id"], tid)
        self.assertEqual(recovered[0]["identity_association"], "nome_visual")
        self.assertFalse(recovered[0]["name_is_ocr"])

    def test_reidentify_after_timeout_only_on_present_observation(self):
        tracker = TargetTracker()
        tid = tracker.update([player()], timestamp_s=0)[0]["track_id"]
        self.assertEqual(tracker.update([], timestamp_s=1.1), [])
        self.assertTrue(
            any(
                x["event"] == "lost"
                for x in tracker.drain_events()
            )
        )
        outcome = tracker.update([player(x=660, y=290)], timestamp_s=1.2)
        self.assertEqual(len(outcome), 1)
        self.assertEqual(outcome[0]["track_id"], tid)
        events = tracker.drain_events()
        self.assertTrue(any(e["event"] == "reidentified" for e in events))

    def test_expired_name_does_not_resurrect_track(self):
        tracker = TargetTracker()
        tid = tracker.update([player()], timestamp_s=0)[0]["track_id"]
        tracker.update([], timestamp_s=7)
        new_id = tracker.update([player()], timestamp_s=7.1)[0]["track_id"]
        self.assertNotEqual(new_id, tid)

    def test_reid_distance_is_bounded(self):
        tracker = TargetTracker()
        tid = tracker.update([player(x=100)], timestamp_s=0)[0]["track_id"]
        new_id = tracker.update([player(x=900)], timestamp_s=0.1)[0]["track_id"]
        self.assertNotEqual(tid, new_id)

    def test_rejected_low_similarity_does_not_force_distant_match(self):
        tracker = TargetTracker()
        tid = tracker.update([player(x=100)], timestamp_s=0)[0]["track_id"]
        second = tracker.update([player(x=500, score=0.35)], timestamp_s=0.1)
        self.assertNotEqual(tid, second[0]["track_id"])
        self.assertEqual(second[0]["identity_association"], "geometria")

    def test_duplicate_name_keys_same_frame_cannot_share_one_id(self):
        tracker = TargetTracker()
        original = tracker.update([player(x=200)], timestamp_s=0)[0]["track_id"]
        both = tracker.update(
            [player(x=202), player(x=900)], timestamp_s=0.1
        )
        self.assertEqual(len(set(o["track_id"] for o in both)), 2)
        self.assertEqual(both[0]["track_id"], original)
        self.assertTrue(all(o["identity_association"] == "geometria" for o in both))

    def test_other_entities_keep_original_geometric_association(self):
        tracker = TargetTracker()
        first = tracker.update([box()], timestamp_s=0)[0]
        second = tracker.update([box(x=645)], timestamp_s=0.1)[0]
        self.assertEqual(first["track_id"], second["track_id"])
        self.assertIsNone(second["track_name_identity_key"])

    def test_reset_never_carries_name_into_next_match(self):
        tracker = TargetTracker()
        tid = tracker.update([player()], timestamp_s=0)[0]["track_id"]
        tracker.reset("resultados_confirmado", timestamp_s=0.5)
        later = tracker.update([player()], timestamp_s=0.6)[0]
        self.assertNotEqual(later["track_id"], tid)

    def test_threshold_is_configurable_and_validated(self):
        with self.assertRaises(ValueError):
            replace(TrackingConfig(), name_reid_min_similarity=1.5)
        with self.assertRaises(ValueError):
            replace(TrackingConfig(), name_reid_max_age_s=0.3)
        tracker = TargetTracker(
            config=replace(TrackingConfig(), name_reid_min_similarity=0.9)
        )
        first = tracker.update([player(x=100, score=0.8)], timestamp_s=0)
        second = tracker.update([player(x=500, score=0.8)], timestamp_s=0.1)
        self.assertNotEqual(first[0]["track_id"], second[0]["track_id"])


if __name__ == "__main__":
    unittest.main()
