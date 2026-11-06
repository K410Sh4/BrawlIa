"""Regressões da segmentação em vegetação e medição de latência ONNX."""

from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from brawlla.config import VisionConfig
from brawlla.health import FrameEvidence
from brawlla.model_vision import OnnxVision
from brawlla.player import PlayerLocator
from brawlla.vision import _bars


def vegetation_frame():
    """Fundo verde conectado ao nome: a busca legada vê um contorno enorme."""
    hsv = np.zeros((720, 1280, 3), dtype=np.uint8)
    hsv[:] = (110, 180, 110)
    hsv[210:450, 300:800] = (60, 230, 165)
    cv2.putText(
        hsv,
        "SkyZynoxCrow",
        (548, 263),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (65, 255, 250),
        2,
    )
    return hsv


def synthetic_bar(_hsv, name, _config, *, evidence=None):
    """Simula somente a confirmação da barra observada, não seu rastreamento."""
    x, y, width, _height = name
    if not (530 <= x <= 680 and 235 <= y <= 270):
        return []
    return [
        {
            "center": [x + width // 2, y + 30],
            "confidence": 0.7,
            "name_region": list(name),
            "_score": 0.7,
        }
    ]


class HighlightNameTests(TestCase):
    def test_bush_discarded_by_regular_path_but_name_survives_highlight(self):
        hsv = vegetation_frame()
        locator = PlayerLocator()
        self.assertEqual(locator._names(hsv), [])
        proposed = locator._names(hsv, highlight=True)
        self.assertTrue(
            any(530 <= item[0][0] <= 680 for item in proposed),
            "A faixa do nome deveria permanecer isolada da vegetação",
        )

    def test_fallback_requires_current_frame_bar_confirmation(self):
        hsv = vegetation_frame()
        locator = PlayerLocator()
        self.assertIsNone(locator.locate(hsv, timestamp_s=1))
        self.assertTrue(locator.diagnostics["highlight_fallback_attempted"])
        self.assertFalse(locator.diagnostics["highlight_fallback_used"])

    def test_fallback_recovers_bush_name_with_independent_bar_evidence(self):
        hsv = vegetation_frame()
        with patch("brawlla.player.bar_observations", side_effect=synthetic_bar):
            locator = PlayerLocator()
            candidate = locator.locate(hsv, timestamp_s=1)
        self.assertIsNotNone(candidate)
        self.assertEqual(locator.diagnostics["search_mode"], "global_highlight")
        self.assertTrue(locator.diagnostics["highlight_fallback_used"])

    def test_fallback_can_be_disabled_for_ab_comparison(self):
        hsv = vegetation_frame()
        locator = PlayerLocator(replace(VisionConfig(), player_highlight_fallback=False))
        with patch("brawlla.player.bar_observations", side_effect=synthetic_bar):
            candidate = locator.locate(hsv, timestamp_s=1)
        self.assertIsNone(candidate)
        self.assertFalse(locator.diagnostics["highlight_fallback_attempted"])

    def test_hsv_floor_validation(self):
        with self.assertRaises(ValueError):
            replace(VisionConfig(), player_highlight_fallback=1)
        with self.assertRaises(ValueError):
            replace(VisionConfig(), player_highlight_hue_min=180)
        with self.assertRaises(ValueError):
            replace(VisionConfig(), player_highlight_value_min=-1)

    def test_cached_green_mask_produces_same_candidate_regions(self):
        hsv = vegetation_frame()
        evidence = FrameEvidence(hsv, VisionConfig())
        expected = _bars(hsv, "green")
        actual = _bars(hsv, "green", color_mask=evidence.color_masks[0][0])
        self.assertEqual(expected, actual)


class OnnxTimingTests(TestCase):
    def test_stage_times_preserve_legacy_total(self):
        class FakeSession:
            def run(self, *_args):
                return [np.array([[[100, 100, 50, 30, 0.95]]], dtype=np.float32)]

        adapter = OnnxVision.__new__(OnnxVision)
        adapter.session = FakeSession()
        adapter.input_name = "images"
        adapter.provider = "CPUExecutionProvider"
        adapter.width = 640
        adapter.height = 640
        adapter.classes = ("player",)
        adapter.export_format = "yolov8"
        result = adapter.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self.assertEqual(result["detections"][0]["label"], "player")
        self.assertGreaterEqual(result["preprocess_ms"], 0)
        self.assertGreaterEqual(result["model_runtime_ms"], 0)
        self.assertGreaterEqual(result["postprocess_ms"], 0)
        self.assertAlmostEqual(
            result["inference_ms"],
            sum(
                result[key]
                for key in ("preprocess_ms", "model_runtime_ms", "postprocess_ms")
            ),
            delta=0.005,
        )
