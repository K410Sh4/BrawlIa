"""Regressões da configuração efetiva, histórico, câmera e dados de treinamento."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import warnings

import cv2
import numpy as np

from brawlla.config import AnalyzerConfig, VisionConfig, SessionOptions, load_config
from brawlla.dataset import (
    save_annotation,
    export_dataset,
    validate_annotation,
)
from brawlla.engine import SessionAnalyzer
from brawlla.evaluation import match_boxes, evaluate_dataset
from brawlla.health import FrameEvidence, fill_masks, number_evidence
from brawlla.observations import ObservationRecorder
from brawlla.provenance import provenance_snapshot
from brawlla.recording import AnalysisRecorder
from brawlla.session import SessionController
from brawlla.tracking import TargetTracker
from test_v09 import sample, obj


def observation(seq):
    return {
        "frame_id": seq,
        "session_id": "synthetic",
        "capture_timestamp_s": float(seq),
        "capture": {"captured_unix": 1700000000 + seq},
        "resolution": [320, 240],
        "game_state": {"name": "desconhecido"},
        "tracked_objects": [],
    }


class ConfigAndEvidenceTests(unittest.TestCase):
    def test_template_matches_effective_defaults(self):
        template = Path(__file__).resolve().parents[1] / "configuracoes.json"
        self.assertEqual(load_config(template), AnalyzerConfig())

    def test_legacy_unused_options_warn_and_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "vision": {
                            "container_dark_fraction": 0.9,
                            "container_dark_support_min": 0.8,
                        }
                    }
                )
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = load_config(path)
            self.assertEqual(len(caught), 2)
            self.assertEqual(config.vision.min_container_fraction, 0.45)
            self.assertNotIn("container_dark_fraction", asdict(config.vision))

    def test_color_parameters_change_active_masks(self):
        hsv = np.array([[[27, 220, 220], [0, 255, 245], [170, 200, 200]]], np.uint8)
        config = VisionConfig()
        original = dict((color, mask) for mask, color in fill_masks(hsv, config))
        self.assertTrue(original["amarela"][0, 0])
        self.assertTrue(original["vermelha"][0, 1])
        self.assertTrue(original["vermelha"][0, 2])
        changed = replace(
            config,
            yellow_lo=(80, 240, 240),
            yellow_hi=(81, 255, 255),
            red_low_lo=(80, 240, 240),
            red_low_hi=(81, 255, 255),
            red_lo=(80, 240, 240),
            red_hi=(81, 255, 255),
        )
        masks = dict((color, mask) for mask, color in fill_masks(hsv, changed))
        self.assertEqual(int(masks["amarela"].max()), 0)
        self.assertEqual(int(masks["vermelha"].max()), 0)

    def test_cache_keeps_local_digit_semantics_and_avoids_duplicate_cc(self):
        hsv = cv2.cvtColor(sample(), cv2.COLOR_BGR2HSV)
        cfg = VisionConfig()
        cache = FrameEvidence(hsv, cfg)
        direct = number_evidence(hsv, 602, 229, 76, config=cfg)
        self.assertEqual(cache.query(602, 229, 76), direct)
        self.assertEqual(cache.query(602, 229, 76), direct)
        self.assertEqual(cache.component_calls, 1)
        self.assertEqual(cache.diagnostics()["number_cache_hits"], 1)

    def test_config_hash_stable_and_changes_with_actual_threshold(self):
        first = provenance_snapshot(AnalyzerConfig())
        same = provenance_snapshot(AnalyzerConfig())
        changed = provenance_snapshot(
            AnalyzerConfig(vision=replace(VisionConfig(), number_value_min=160))
        )
        self.assertEqual(first["config_sha256"], same["config_sha256"])
        self.assertNotEqual(first["config_sha256"], changed["config_sha256"])
        self.assertIn("numpy", first["packages"])

    def test_player_uses_local_region_but_falls_back_to_full_search(self):
        engine = SessionAnalyzer()
        frame = sample()
        hud = engine.states.hud
        frame[20 : 20 + hud.shape[0], 40 : 40 + hud.shape[1]] = hud
        engine.analyze(frame, 0)
        _, second = engine.analyze(frame, 0.067)
        self.assertEqual(second["player_detection_diagnostics"]["search_mode"], "local")
        self.assertIsNotNone(second["player_health_bar_candidate"])
        blank = np.zeros_like(frame)
        blank[20 : 20 + hud.shape[0], 40 : 40 + hud.shape[1]] = hud
        _, third = engine.analyze(blank, 0.134)
        self.assertIsNone(third["player_health_bar_candidate"])
        self.assertEqual(
            third["player_detection_diagnostics"]["search_mode"], "local_then_global"
        )


class CameraAssociationTests(unittest.TestCase):
    def test_pan_larger_than_gate_preserves_id(self):
        tracker = TargetTracker()
        first = tracker.update([obj(200)], 1280, 0)[0]
        camera = {
            "valid": True,
            "affine_original": [[1, 0, 250], [0, 1, 0]],
            "resolution": [1280, 720],
        }
        second = tracker.update([obj(450)], 1280, 0.1, camera_motion=camera)[0]
        self.assertEqual(first["track_id"], second["track_id"])
        self.assertEqual(second["association_reference"], "camera_compensated")
        self.assertAlmostEqual(second["association_motion_px_per_s"][0], 0)
        self.assertAlmostEqual(second["screen_motion_px_per_s"][0], 2500)

    def test_camera_projection_accumulates_across_missing_observations(self):
        tracker = TargetTracker()
        tid = tracker.update([obj(200)], 1280, 0)[0]["track_id"]
        camera = {"valid": True, "affine_original": [[1, 0, 100], [0, 1, 0]]}
        self.assertEqual(tracker.update([], 1280, 0.1, camera_motion=camera), [])
        result = tracker.update([obj(400)], 1280, 0.2, camera_motion=camera)[0]
        self.assertEqual(result["track_id"], tid)

    def test_invalid_camera_matrix_is_not_applied(self):
        tracker = TargetTracker()
        tid = tracker.update([obj(200)], 1280, 0)[0]["track_id"]
        result = tracker.update(
            [obj(201)],
            1280,
            0.1,
            camera_motion={
                "valid": True,
                "affine_original": [[float("nan"), 0, 0], [0, 1, 0]],
            },
        )[0]
        self.assertEqual(result["track_id"], tid)
        self.assertEqual(result["association_reference"], "screen")

    def test_engine_separates_camera_association_and_body_timings(self):
        engine = SessionAnalyzer()
        frame = sample()
        hud = engine.states.hud
        frame[20 : 20 + hud.shape[0], 40 : 40 + hud.shape[1]] = hud
        _, result = engine.analyze(frame, 0)
        for key in ("camera_motion", "association", "body_motion"):
            self.assertIn(key, result["timings_ms"])
        self.assertEqual(
            result["timings_ms"]["tracking"], result["timings_ms"]["association"]
        )


class HistoryTests(unittest.TestCase):
    def test_every_accepted_observation_is_written_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = ObservationRecorder(tmp)
            history.start()
            for seq in range(1, 41):
                history.submit(observation(seq))
            history.stop()
            history.join(5)
            rows = [
                json.loads(line) for line in history.log_path.read_text().splitlines()
            ]
            self.assertEqual([row["frame_id"] for row in rows], list(range(1, 41)))
            self.assertTrue(history.snapshot()["complete"])
            self.assertEqual(history.metrics.frames, 40)

    def test_overflow_is_explicit_never_silent_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = ObservationRecorder(tmp, queue_capacity=1)
            history.submit(observation(1))
            with self.assertRaisesRegex(RuntimeError, "Fila"):
                history.submit(observation(2))
            self.assertEqual(history.snapshot()["accepted"], 1)
            self.assertEqual(history.snapshot()["rejected"], 1)
            history.start()
            history.join(5)
            self.assertEqual(history.records, 1)
            self.assertFalse(history.snapshot()["complete"])

    def test_byte_limit_reports_incomplete_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = ObservationRecorder(tmp, max_bytes=1)
            history.start()
            history.submit(observation(1))
            history.stop()
            history.join(5)
            self.assertIn("Limite", history.error)
            self.assertEqual(history.records, 0)
            self.assertFalse(history.snapshot()["complete"])

    def test_slow_video_does_not_drop_history_and_config_is_saved(self):
        ready = threading.Event()
        release = threading.Event()
        outputs = []

        class Source:
            def grab(self):
                return np.full((240, 320, 3), 90, np.uint8)

            def close(self):
                pass

        original = AnalysisRecorder._open_writer

        def slow(recorder, size):
            release.wait(4)
            return original(recorder, size)

        def notify(kind, payload):
            if kind == "frame":
                outputs.append(payload[1])
                if len(outputs) >= 8:
                    ready.set()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(AnalysisRecorder, "_open_writer", slow),
        ):
            controller = SessionController(notify, root=tmp)
            controller.start(Source, SessionOptions(record_video=True))
            try:
                self.assertTrue(ready.wait(4))
            finally:
                controller.stop()
                release.set()
                self.assertTrue(controller.join(6))
            audit = json.loads((controller.folder / "sessao.json").read_text())
            rows = [
                json.loads(line)
                for line in (controller.folder / "analise.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), audit["pipeline"]["analyzed"])
            self.assertEqual(audit["history"]["rejected"], 0)
            self.assertTrue(audit["history"]["complete"])
            self.assertGreater(audit["video"]["dropped"], 0)
            self.assertEqual(
                audit["analyzer_config"],
                json.loads(json.dumps(asdict(AnalyzerConfig()))),
            )
            self.assertEqual(
                audit["provenance"]["config_sha256"],
                provenance_snapshot(AnalyzerConfig())["config_sha256"],
            )
            summary = json.loads((controller.folder / "resumo.json").read_text())
            self.assertEqual(summary["events_recorded"], len(rows))
            self.assertEqual(summary["events_file"], "analise.jsonl")
            self.assertTrue((controller.folder / "video_frames.jsonl").exists())

    def test_history_exists_with_video_disabled(self):
        ready = threading.Event()

        class Source:
            def grab(self):
                return np.full((240, 320, 3), 90, np.uint8)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            controller = SessionController(
                lambda kind, payload: ready.set() if kind == "frame" else None, root=tmp
            )
            controller.start(Source, SessionOptions(record_video=False))
            try:
                self.assertTrue(ready.wait(3))
            finally:
                controller.stop()
                self.assertTrue(controller.join(5))
            audit = json.loads((controller.folder / "sessao.json").read_text())
            self.assertEqual(audit["history"]["records"], audit["pipeline"]["analyzed"])
            self.assertTrue((controller.folder / "analise.jsonl").exists())
            self.assertFalse(list(controller.folder.glob("*.mp4")))


class DatasetRelationsTests(unittest.TestCase):
    def test_invalid_owner_and_duplicate_entity_are_rejected(self):
        for boxes in [
            [{"category": "barra_vida", "bbox": [1, 1, 10, 5], "owner_id": "missing"}],
            [
                {"category": "jogador", "bbox": [1, 1, 20, 20], "entity_id": "a"},
                {"category": "caixa", "bbox": [40, 40, 20, 20], "entity_id": "a"},
            ],
        ]:
            with self.assertRaises(ValueError):
                validate_annotation({"status": "corrected", "boxes": boxes}, 100, 100)

    def test_export_preserves_bar_owner_relation_separately_from_coco(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "game"
            folder.mkdir()
            entry = {
                "image": "frame_1.png",
                "prediction": "frame_1.prediction.json",
                "width": 320,
                "height": 240,
                "session_id": "game",
                "frame_id": 1,
            }
            cv2.imwrite(
                str(folder / entry["image"]), np.full((240, 320, 3), 90, np.uint8)
            )
            (folder / "manifest.jsonl").write_text(json.dumps(entry) + "\n")
            boxes = [
                {"category": "jogador", "bbox": [100, 80, 40, 60], "entity_id": "hero"},
                {
                    "category": "barra_vida",
                    "bbox": [100, 60, 40, 5],
                    "owner_id": "hero",
                },
            ]
            save_annotation(folder, entry, {"status": "corrected", "boxes": boxes})
            summary = export_dataset(root, root / "export")
            relation = json.loads((root / "export/relationships.json").read_text())[0]
            self.assertEqual(relation["owner_annotation_id"], 1)
            self.assertEqual(relation["bar_annotation_id"], 2)
            self.assertEqual(summary["unlinked_bars"], 0)
            self.assertEqual(summary["relationships"], 1)

    def test_evaluation_matching_counts_duplicates_as_false_positive(self):
        truth = [[10, 10, 20, 5]]
        self.assertEqual(
            match_boxes([[10, 10, 20, 5], [10, 10, 20, 5]], truth), (1, 1, 0)
        )
        self.assertEqual(match_boxes([], truth), (0, 0, 1))

    def test_evaluation_ignores_unreviewed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.jsonl").write_text("")
            result = evaluate_dataset(root)
            self.assertEqual(result["frames"], 0)
            self.assertIsNone(result["precision"])
