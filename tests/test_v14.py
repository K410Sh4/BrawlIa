"""V1.4: terreno temporizado, recuperação do movimento, OCR e captura opcionais."""

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from brawlla.capture import AutoWindowCapture, GameWindow
from brawlla.config import RosterConfig, VisionConfig
from brawlla.roster import MatchRoster
from brawlla.terrain import TerrainObserver
from brawlla.vision import detect


def synthetic_terrain():
    hsv = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(hsv, (70, 110), (190, 230), (94, 240, 165), -1)
    cv2.rectangle(hsv, (280, 100), (440, 200), (168, 100, 225), -1)
    return hsv


class TerrainCacheTests(unittest.TestCase):
    def test_reuses_cache_and_refreshes_after_limit(self):
        config = VisionConfig()
        terrain = TerrainObserver(config)
        hsv = synthetic_terrain()
        one, _, first = terrain.observe(hsv, 0.0)
        self.assertTrue(first["refreshed"])
        self.assertTrue(one["bush"])
        two, _, reused = terrain.observe(hsv, 0.033)
        self.assertFalse(reused["refreshed"])
        self.assertEqual(one, two)
        self.assertTrue(reused["approximate"])
        _, _, third = terrain.observe(hsv, 0.066)
        self.assertFalse(third["refreshed"])
        _, _, fourth = terrain.observe(hsv, 0.099)
        self.assertTrue(fourth["refreshed"])
        self.assertEqual(fourth["refreshes"], 2)
        self.assertEqual(fourth["reuses"], 2)

    def test_camera_transform_shifts_cached_regions(self):
        cache = TerrainObserver(VisionConfig())
        hsv = synthetic_terrain()
        original, _, _ = cache.observe(hsv, 0)
        camera = {
            "valid": True,
            "affine_original": [[1, 0, 14], [0, 1, -4]],
        }
        result, _, diag = cache.observe(hsv, 0.033, camera)
        self.assertTrue(diag["motion_compensated"])
        self.assertEqual(result["bush"][0].x, original["bush"][0].x + 14)
        self.assertEqual(result["bush"][0].y, original["bush"][0].y - 4)

    def test_resets_after_resolution_change_and_config_validation(self):
        cache = TerrainObserver(VisionConfig())
        cache.observe(synthetic_terrain(), 0)
        _, _, diag = cache.observe(np.zeros((720, 1280, 3), np.uint8), 0.01)
        self.assertEqual(diag["refreshes"], 1)
        with self.assertRaises(ValueError):
            replace(VisionConfig(), terrain_sample_every=0)
        with self.assertRaises(ValueError):
            replace(VisionConfig(), terrain_cache_max_s=0.6)

    def test_full_detector_reports_stage_durations(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        data, _ = detect(frame)
        self.assertIn("vision_terrain", data["vision_stage_timings_ms"])
        self.assertIn("vision_other_bars", data["vision_stage_timings_ms"])
        self.assertTrue(data["terrain_sampling"]["refreshed"])


class OCRDiagnosticsTests(unittest.TestCase):
    def test_records_empty_low_confidence_and_valid_reads(self):
        roster = MatchRoster(RosterConfig())
        roster.detected = True
        reads = [("", 0.0), ("Nick2", 0.3), ("Nick3", 0.92)] + [("", 0.0)] * 7
        roster.completed.put((roster.generation, "roster", reads, 40.5))
        roster._poll()
        diag = roster.snapshot()["ocr_diagnostics"]
        self.assertEqual(diag["finished_roster_batches"], 1)
        self.assertEqual(diag["per_slot"][0]["empty"], 1)
        self.assertEqual(diag["per_slot"][1]["low_confidence"], 1)
        self.assertEqual(diag["per_slot"][2]["accepted"], 1)
        self.assertEqual(diag["recent_batch_ms"], [40.5])
        self.assertEqual(roster.snapshot()["confirmed_names"], 0)

    def test_probes_intro_at_limited_frequency(self):
        roster = MatchRoster(RosterConfig())
        frame = np.zeros((720, 1280, 3), np.uint8)
        with patch("brawlla.roster.roster_visible", return_value=False) as spy:
            for timestamp in (0, 0.03, 0.07, 0.12, 0.17, 0.19):
                roster.observe(frame, "desconhecido", timestamp, [])
        self.assertEqual(spy.call_count, 2)


class CaptureBackendTests(unittest.TestCase):
    def test_dxcam_opt_in_and_safe_mss_fallback(self):
        window = GameWindow(42, "BlueStacks", 640, 360)
        frame = np.zeros((360, 640, 3), np.uint8)
        with (
            patch.dict(os.environ, {"BRAWLIA_CAPTURE_BACKEND": "dxcam"}),
            patch("brawlla.capture.list_game_windows", return_value=[window]),
            patch("brawlla.capture.DxcamWindowCapture") as dx,
            patch("brawlla.capture.WindowCapture") as mss,
        ):
            dx.return_value.grab.return_value = frame
            dx.return_value._last_rect = [0, 0, 640, 360]
            dx.return_value.quality_flags = []
            source = AutoWindowCapture()
            self.assertEqual(source.backend, "dxcam")
            self.assertIs(source.grab(), frame)
            dx.return_value.grab.side_effect = RuntimeError("GPU unavailable")
            mss.return_value.grab.return_value = frame
            mss.return_value._last_rect = [0, 0, 640, 360]
            mss.return_value.quality_flags = []
            self.assertIs(source.grab(), frame)
            self.assertEqual(source.backend, "mss")
            dx.return_value.close.assert_called_once()
            source.close()

    def test_missing_dxcam_falls_back_during_source_initialization(self):
        window = GameWindow(42, "BlueStacks", 640, 360)
        with (
            patch.dict(os.environ, {"BRAWLIA_CAPTURE_BACKEND": "dxcam"}),
            patch("brawlla.capture.list_game_windows", return_value=[window]),
            patch("brawlla.capture.DxcamWindowCapture", side_effect=ImportError()),
            patch("brawlla.capture.WindowCapture"),
        ):
            source = AutoWindowCapture()
            self.assertEqual(source.backend, "mss")
            source.close()


if __name__ == "__main__":
    unittest.main()
