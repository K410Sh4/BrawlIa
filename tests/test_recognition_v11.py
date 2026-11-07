"""Reconhecimento V1.1: regressões de estados, evidência e transição de cena."""

import unittest

import numpy as np

from brawlla.recognition_v1.game_state_memory import GameStateMemory
from brawlla.recognition_v1.hud_context import HUDAnchors
from brawlla.recognition_v1.recognizer import RecognitionV1, RegionImage
from brawlla.recognition_v1.geometry import Rect
from brawlla.recognition_v1.roi_manager import ROIManager
from brawlla.recognition_v1.scene_change import SceneTransitionMonitor, scene_probe_rect


class MemoryStateTests(unittest.TestCase):
    def test_terminal_state_survives_unchecked_rois(self):
        memory = GameStateMemory()
        memory.update(None, 1.0, "combate", 0.95)
        memory.update(None, 3.0, "derrota", 0.90)
        memory.update(None, 4.0, "resultados", 0.92)
        for n in range(1, 80):
            result = memory.update(None, 4.0 + n / 15)
            self.assertEqual(result["name"], "resultados")
            self.assertEqual(result["evidence"]["visual"], "NOT_CHECKED")
            self.assertEqual(result["evidence"]["hud"], "NOT_CHECKED")
        self.assertEqual(
            memory.update(HUDAnchors(True, True), 10.0)["name"], "resultados"
        )
        self.assertEqual(
            memory.update(None, 10.5, "desconhecido")["name"], "resultados"
        )

    def test_terminal_confidence_depends_on_elapsed_time_not_frame_count(self):
        fast = GameStateMemory()
        slow = GameStateMemory()
        fast.update(None, 1.0, "resultados", 0.92)
        slow.update(None, 1.0, "resultados", 0.92)
        for n in range(1, 61):
            fast.update(None, 1.0 + n / 30.0)
        slow.update(None, 3.0)
        self.assertEqual(fast.confidence, slow.confidence)
        self.assertGreater(fast.confidence, 0.75)
        self.assertEqual(fast.update(None, 3.0)["name"], "resultados")
        self.assertAlmostEqual(fast.confidence, 0.92 * 0.93 ** 2, places=7)

    def test_terminal_reconfirmation_restores_confidence_anchor(self):
        memory = GameStateMemory()
        memory.update(None, 1.0, "resultados", 0.9)
        memory.update(None, 10.0)
        self.assertLess(memory.confidence, 0.9)
        memory.update(None, 10.1, "resultados", 0.95)
        self.assertAlmostEqual(memory.confidence, 0.95)
        memory.update(None, 11.1)
        self.assertAlmostEqual(memory.confidence, 0.95 * 0.93, places=7)

    def test_exit_terminal_requires_two_distinct_global_confirms(self):
        memory = GameStateMemory()
        memory.update(None, 1.0, "resultados", 0.91)
        first = memory.update(None, 2.0, "pre_partida", 0.9)
        self.assertEqual(first["name"], "resultados")
        self.assertEqual(first["reason"], "transicao_pendente")
        self.assertEqual(memory.update(None, 2.1)["name"], "resultados")
        self.assertEqual(
            memory.update(None, 2.5, "pre_partida", 0.9)["name"], "pre_partida"
        )

    def test_interrupted_terminal_exit_does_not_accumulate_false_votes(self):
        memory = GameStateMemory()
        memory.update(None, 0.0, "resultados", 0.9)
        memory.update(None, 1.0, "combate", 0.9)
        memory.update(None, 1.1, "desconhecido", 0.0)
        self.assertEqual(memory.update(None, 2.0, "combate", 0.9)["name"], "resultados")

    def test_skipped_hud_is_not_negative_hud(self):
        memory = GameStateMemory()
        yes = HUDAnchors(True, True)
        first = memory.update(yes, 1.0)
        self.assertEqual(first["name"], "desconhecido")
        self.assertEqual(memory.update(None, 1.1)["evidence"]["hud"], "NOT_CHECKED")
        self.assertEqual(memory.update(yes, 1.3)["name"], "combate")
        self.assertEqual(memory.update(HUDAnchors(), 2.0)["evidence"]["hud"], "ABSENT")


class FakeResults:
    def detect(self, frame, timestamp_s=None):
        return {"name": "resultados", "confidence": 0.94}


class RecognitionStateTests(unittest.TestCase):
    def test_results_never_flap_on_regional_frames(self):
        detector = RecognitionV1(state_detector=FakeResults())
        size = (640, 360)
        full = np.zeros((360, 640, 3), dtype=np.uint8)
        first = detector.observe(
            [RegionImage("full", Rect(0, 0, 640, 360), full)],
            size,
            now=1.0,
            full=True,
        )
        self.assertEqual(first["state"]["name"], "resultados")
        plan = ROIManager()
        plan.note_full(1.0)
        self.assertEqual(
            plan.plan(1.1, size, "resultados", []).reason, "estado_terminal_regional"
        )
        probe = scene_probe_rect(size)
        for n in range(1, 60):
            t = 1.0 + n / 15.0
            region = RegionImage(
                "scene_probe",
                probe,
                full[probe.y : probe.y + probe.h, probe.x : probe.x + probe.w].copy(),
            )
            result = detector.observe([region], size, now=t, full=False)
            self.assertEqual(result["state"]["name"], "resultados")
            self.assertFalse(
                any(e.get("event") == "state_change" for e in result["events"])
            )


class SceneChangeTests(unittest.TestCase):
    def test_two_large_differences_request_full_scan(self):
        size = (640, 360)
        rect = scene_probe_rect(size)
        black = np.zeros((rect.h, rect.w, 3), dtype=np.uint8)
        white = np.full_like(black, 255)
        monitor = SceneTransitionMonitor(change_votes=2)
        self.assertFalse(
            monitor.observe(
                [RegionImage("scene_probe", rect, black)],
                size,
                "resultados",
                0,
                full=False,
            )
        )
        self.assertFalse(
            monitor.observe(
                [RegionImage("scene_probe", rect, white)],
                size,
                "resultados",
                0.1,
                full=False,
            )
        )
        self.assertTrue(
            monitor.observe(
                [RegionImage("scene_probe", rect, white)],
                size,
                "resultados",
                0.2,
                full=False,
            )
        )
        self.assertFalse(
            monitor.observe(
                [RegionImage("scene_probe", rect, white)],
                size,
                "combate",
                0.3,
                full=False,
            )
        )


class GlobalAssociationTests(unittest.TestCase):
    def test_global_optimum_outperforms_greedy_confidence(self):
        from brawlla.recognition_v1.association import optimal_associations

        # Linha 0 disputa track 0 com linha 1; o ótimo respeita ambas.
        scores = [[0.91, 0.88], [0.85, 0.05]]
        self.assertEqual(optimal_associations(scores), [(0, 1), (1, 0)])

    def test_unmatched_observations_cannot_force_false_identity(self):
        from brawlla.recognition_v1.association import optimal_associations

        self.assertEqual(
            optimal_associations(
                [
                    [float("-inf"), 0.03],
                    [0.75, float("-inf")],
                ]
            ),
            [(1, 0)],
        )


class DatasetCuratorTests(unittest.TestCase):
    def test_duplicates_are_suppressed_but_events_stay(self):
        from brawlla.recognition_v1.dataset_curator import DatasetCurator

        curator = DatasetCurator(interval_s=0.5)
        raw = np.zeros((32, 64, 3), dtype=np.uint8)
        result = {"timestamp_s": 1.0, "state": {"name": "combate"}, "events": []}
        self.assertEqual(curator.select_samples(result), (True, "periodic"))
        key = ("scene", (0, 0, 64, 32), (640, 360), "combate")
        self.assertFalse(curator.detect_duplicate_samples(raw, key))
        self.assertTrue(curator.detect_duplicate_samples(raw, key))
        bright = np.full_like(raw, 255)
        self.assertFalse(curator.detect_duplicate_samples(bright, key))
        self.assertFalse(curator.detect_duplicate_samples(raw, key, force=True))
        event = {
            "timestamp_s": 1.1,
            "events": [{"event": "state_change"}],
            "state": {"name": "resultados"},
        }
        self.assertEqual(curator.select_samples(event), (True, "critical_event"))

    def test_unverified_predictions_are_not_training_labels(self):
        from brawlla.recognition_v1.dataset_curator import DatasetCurator

        self.assertTrue(
            DatasetCurator.validate_annotations(
                {
                    "label_status": "unreviewed",
                    "prediction": [{"track_id": 1}],
                }
            )
        )
        self.assertFalse(
            DatasetCurator.validate_annotations(
                {
                    "label_status": "accepted",
                    "prediction": [{"track_id": 1}],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
