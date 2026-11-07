"""Regression tests for audit P0/P1: semantic ROI, budgets and evidence TTL."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import cv2
import numpy as np

from brawlla.recognition_v1.geometry import CaptureRegion, Rect
from brawlla.recognition_v1.roi_manager import ROIManager
from brawlla.recognition_v1.context_engine import ContextMemory, Detection
from brawlla.recognition_v1.recognizer import RecognitionV1, RegionImage
from brawlla.recognition_v1.dataset_store import DatasetStore


class CombatOnly:
    def detect(self, frame, timestamp_s=None):
        return {"name": "combate", "confidence": 0.99}


class ROIBudgetTests(unittest.TestCase):
    def test_merged_regions_keep_all_original_roles(self):
        hud = CaptureRegion(
            "hud_right", Rect(768, 432, 512, 288), "button", 2
        )
        target = CaptureRegion(
            "target_7", Rect(773, 452, 224, 183), "player", 3
        )
        merged = ROIManager._merge([hud, target])
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].has_target)
        self.assertEqual(
            {role for role, rect in merged[0].source_regions},
            {"hud_right", "target_7"},
        )
        self.assertEqual(
            merged[0].rect.xywh(), [768, 432, 512, 288]
        )

    def test_many_tracks_are_budgeted_not_full_screen(self):
        manager = ROIManager(full_interval_s=1.0, max_targets=3)
        manager.note_full(0.0)
        tracks = [
            {
                "track_id": n,
                "status": "observado",
                "hits": 5,
                "confirmed": True,
                "color": "verde" if n % 2 else "vermelha",
                "kind": "barra_vida_candidata",
                "confidence": 0.8,
                "xywh": [100 + (n % 12) * 80, 110 + (n % 6) * 44, 65, 7],
                "velocity": [0, 0],
                "last_seen": 0.0,
            }
            for n in range(50)
        ]
        plan = manager.plan(0.2, (1280, 720), "combate", tracks)
        self.assertFalse(plan.full)
        self.assertLessEqual(plan.coverage, 0.55)
        self.assertGreaterEqual(plan.suppressed_targets, 47)
        self.assertTrue(manager.plan(1.1, (1280, 720), "combate", tracks).full)

    def test_off_combat_does_not_capture_full_every_tick(self):
        manager = ROIManager(full_interval_s=1.0)
        manager.note_full(0.0)
        plan = manager.plan(0.1, (1280, 720), "pre_partida", [])
        self.assertFalse(plan.full)
        self.assertEqual(plan.reason, "estado_estavel_regional")
        self.assertLess(plan.coverage, 0.20)


class SemanticDetectionTests(unittest.TestCase):
    def test_merged_hud_target_retains_bar_detection(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(frame, (850, 500), (920, 507), (45, 220, 45), -1)
        detector = RecognitionV1(state_detector=CombatOnly())
        full = RegionImage("full", Rect(0, 0, 1280, 720), frame)
        initial = detector.observe([full], (1280, 720), now=1.0, full=True)
        self.assertGreater(initial["candidates_in_frame"], 0)
        merged = ROIManager._merge([
            CaptureRegion(
                "hud_right", Rect(768, 432, 512, 288), "hud", 2
            ),
            CaptureRegion(
                "target_1", Rect(773, 452, 224, 183), "target", 3
            ),
        ])[0]
        rect = merged.rect
        crop = frame[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w]
        region = RegionImage(
            merged.name, rect, crop, sources=merged.source_regions
        )
        again = detector.observe(
            [region], (1280, 720), now=1.3, full=False
        )
        self.assertGreater(again["candidates_in_frame"], 0)
        self.assertTrue(any(t["confirmed"] for t in again["tracks"]))
        self.assertTrue(any(
            item["role"] == "target_1"
            for item in again["regions"][0]["sources"]
        ))


class CandidateLifecycleTests(unittest.TestCase):
    @staticmethod
    def detection():
        return Detection(
            Rect(200, 100, 60, 7),
            "barra_vida_candidata", "verde", 0.75,
        )

    def test_candidates_do_not_become_priority_targets_until_confirmed(self):
        memory = ContextMemory()
        first = memory.update(
            [self.detection()], [Rect(0, 0, 640, 360)], 0, (640, 360)
        )
        self.assertEqual(first[0]["status"], "candidate")
        self.assertFalse(first[0]["confirmed"])
        second = memory.update(
            [self.detection()], [Rect(0, 0, 640, 360)], 0.10, (640, 360)
        )
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])
        self.assertTrue(second[0]["confirmed"])

    def test_lost_even_if_previous_region_not_observed(self):
        memory = ContextMemory(max_absence_s=0.9, candidate_absence_s=0.25)
        memory.update(
            [self.detection()], [Rect(0, 0, 640, 360)], 0, (640, 360)
        )
        expired = memory.update([], [], 0.3, (640, 360))
        self.assertFalse(expired)
        memory.update(
            [self.detection()], [Rect(0, 0, 640, 360)], 1.0, (640, 360)
        )
        memory.update(
            [self.detection()], [Rect(0, 0, 640, 360)], 1.1, (640, 360)
        )
        self.assertFalse(memory.update([], [], 2.1, (640, 360)))


class StorageTests(unittest.TestCase):
    def test_candidate_events_do_not_trigger_png_storm(self):
        with TemporaryDirectory() as tmp:
            recorder = DatasetStore(
                Path(tmp) / "session", min_free_bytes=0,
                sample_interval_s=0.5,
            )
            recorder.start()
            image = np.zeros((32, 64, 3), np.uint8)
            for idx, stamp in enumerate((0.0, 0.05, 0.10), start=1):
                result = {
                    "frame_id": idx, "timestamp_s": stamp,
                    "screen_resolution": [640, 360],
                    "capture_mode": "regional", "tracks": [],
                    "events": [{"event": "candidate", "track_id": idx}],
                }
                region = RegionImage(
                    "target_1", Rect(100, 100, 64, 32), image,
                    captured_at=stamp + 0.005,
                )
                recorder.submit(result, [region])
            recorder.stop()
            recorder.join(5)
            self.assertIsNone(recorder.error)
            self.assertEqual(recorder.records, 3)
            self.assertEqual(recorder.written, 1)
            entries = (Path(tmp) / "session" / "manifest.jsonl").read_text()
            first = json.loads(entries)
            self.assertAlmostEqual(first["captured_at_s"], 0.005)


if __name__ == "__main__":
    unittest.main()
