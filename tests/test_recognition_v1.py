"""Regressões determinísticas da nova aquisição regional e memória visual."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import cv2
import numpy as np

from brawlla.recognition_v1.geometry import Rect
from brawlla.recognition_v1.silhouette import structural_change_ratio
from brawlla.recognition_v1.hud_context import HUDContext, HUDAnchors
from brawlla.recognition_v1.context_engine import ContextMemory, Detection
from brawlla.recognition_v1.roi_manager import ROIManager
from brawlla.recognition_v1.recognizer import RecognitionV1, RegionImage
from brawlla.recognition_v1.dataset_store import DatasetStore


class FakeCombat:
    def detect(self, frame, timestamp_s=None):
        return {"name": "combate", "confidence": 0.95}


class RectTests(unittest.TestCase):
    def test_clips_global_coordinates_and_merges(self):
        a = Rect(90, 90, 30, 30)
        self.assertEqual(a.clipped((100, 100)).xywh(), [90, 90, 10, 10])
        self.assertEqual(a.merged(Rect(100, 100, 20, 20)).xywh(), [90, 90, 30, 30])
        self.assertIsNone(a.translated(200, 0, (100, 100)))
        with self.assertRaises(ValueError):
            Rect(0, 0, 0, 10)


class SilhouetteTests(unittest.TestCase):
    def test_numbers_excluded_from_stable_mask(self):
        before = np.zeros((30, 100, 3), np.uint8)
        after = before.copy()
        after[0:8, 10:80] = (255, 255, 255)
        mask = np.ones((30, 100), np.uint8)
        mask[0:8] = 0
        self.assertEqual(structural_change_ratio(before, after, mask), 0)

    def test_large_pixel_change_is_not_identity_proof(self):
        before = np.zeros((40, 80, 3), np.uint8)
        after = np.full_like(before, 255)
        mask = np.ones((40, 80), np.uint8)
        self.assertEqual(structural_change_ratio(before, after, mask), 1)


class HUDTests(unittest.TestCase):
    def test_controls_preserve_context_during_occlusion(self):
        hud = HUDContext(grace_s=2.5)
        anchors = HUDAnchors(attack=True, movement=True)
        hud.update(anchors, 1.0)
        self.assertEqual(hud.update(anchors, 1.1)["name"], "combate")
        hidden = hud.update(HUDAnchors(), 2.0)
        self.assertEqual(hidden["name"], "combate")
        self.assertTrue(hidden["uncertain"])
        self.assertEqual(hud.update(HUDAnchors(), 4.0)["name"], "desconhecido")
        done = hud.update(HUDAnchors(), 4.1, visual_state="resultados")
        self.assertEqual(done["name"], "resultados")


class MemoryTests(unittest.TestCase):
    def test_not_observed_is_not_occluded(self):
        memory = ContextMemory(max_absence_s=1.5)
        det = Detection(Rect(200, 100, 70, 7), "barra_vida_candidata",
                        "verde", 0.7)
        tracks = memory.update([det], [Rect(0, 0, 640, 360)], 0.0, (640, 360))
        track_id = tracks[0]["track_id"]
        tracks = memory.update([], [Rect(0, 250, 100, 100)], 0.2, (640, 360))
        self.assertEqual(tracks[0]["status"], "nao_observado")
        tracks = memory.update([], [Rect(180, 80, 130, 60)], 0.3, (640, 360))
        self.assertEqual(tracks[0]["status"], "ocluido_ou_nao_detectado")
        self.assertEqual(tracks[0]["track_id"], track_id)

    def test_health_change_preserves_identity(self):
        memory = ContextMemory()
        first = Detection(Rect(200, 100, 70, 8), "barra_vida_candidata",
                          "verde", 0.70)
        second = Detection(Rect(202, 100, 55, 8), "barra_vida_candidata",
                           "verde", 0.80)
        a = memory.update([first], [Rect(0, 0, 640, 360)], 1.0, (640, 360))
        b = memory.update([second], [Rect(0, 0, 640, 360)], 1.1, (640, 360))
        self.assertEqual(a[0]["track_id"], b[0]["track_id"])
        self.assertIsNone(b[0]["health_exact"])


class ROIPlanTests(unittest.TestCase):
    def test_periodic_global_scan_and_dynamic_roi(self):
        policy = ROIManager(full_interval_s=1.0)
        self.assertTrue(policy.plan(0, (1280, 720), "combate", []).full)
        policy.note_full(0.0)
        track = {
            "track_id": 3, "status": "observado",
            "xywh": [600, 260, 70, 8], "last_seen": 0.0,
            "velocity": [100, 0],
        }
        plan = policy.plan(0.2, (1280, 720), "combate", [track])
        self.assertFalse(plan.full)
        self.assertLess(plan.coverage, 0.55)
        self.assertTrue(any("target" in r.name for r in plan.regions))
        self.assertTrue(policy.plan(1.1, (1280, 720), "combate", [track]).full)
        policy.request_full_scan("camera_perdida")
        self.assertEqual(policy.plan(0.4, (1280, 720), "combate", [track]).reason,
                         "camera_perdida")


class DetectorTests(unittest.TestCase):
    def test_full_and_regional_share_global_coordinates(self):
        full = np.zeros((360, 640, 3), np.uint8)
        cv2.rectangle(full, (200, 100), (260, 105), (45, 220, 45), -1)
        detector = RecognitionV1(state_detector=FakeCombat())
        first = detector.observe(
            [RegionImage("full", Rect(0, 0, 640, 360), full)],
            (640, 360), now=1.0, full=True,
        )
        self.assertTrue(first["tracks"])
        cropped = full[80:160, 180:300].copy()
        second = detector.observe(
            [RegionImage("target_1", Rect(180, 80, 120, 80), cropped)],
            (640, 360), now=1.1, full=False,
        )
        self.assertTrue(second["tracks"])
        self.assertEqual(first["tracks"][0]["track_id"],
                         second["tracks"][0]["track_id"])
        self.assertGreaterEqual(second["tracks"][0]["xywh"][0], 190)


class DatasetTests(unittest.TestCase):
    def test_dataset_saves_raw_png_and_metadata(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "session"
            recorder = DatasetStore(folder, min_free_bytes=0)
            recorder.start()
            image = np.zeros((40, 80, 3), np.uint8)
            result = {
                "frame_id": 1, "timestamp_s": 1.0,
                "screen_resolution": [640, 360],
                "capture_mode": "regional",
                "tracks": [], "events": [],
            }
            recorder.submit(result, [
                RegionImage("target", Rect(100, 100, 80, 40), image)
            ])
            recorder.stop()
            recorder.join(5)
            self.assertIsNone(recorder.error)
            self.assertEqual(recorder.written, 1)
            entry = json.loads((folder / "manifest.jsonl").read_text())
            self.assertFalse(entry["ground_truth"])
            self.assertEqual(entry["roi_xywh"], [100, 100, 80, 40])
            self.assertTrue((folder / entry["image"]).exists())


if __name__ == "__main__":
    unittest.main()
