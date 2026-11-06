"""Regressões de integridade: dados brutos, geometria, revisão e isolamento."""

import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from brawlla.capture import WindowCapture
from brawlla.config import SessionOptions
from brawlla.dataset import (
    crop_frame,
    QualityMonitor,
    save_annotation,
    load_entries,
    export_dataset,
)
from brawlla.domain import FramePacket
from brawlla.recording import AnalysisRecorder, fit_video_frame
from brawlla.samples import SampleRecorder
from brawlla.session import SessionController
from brawlla.storage import load_preferences, save_preferences, write_json
from brawlla.stream import VisionPipeline


def packet(seq, now, frame=None, **metadata):
    if frame is None:
        frame = np.full((240, 320, 3), 90, np.uint8)
    return FramePacket(
        seq, now, frame, metadata={"captured_unix": 1700000000 + now, **metadata}
    )


def data():
    return {
        "resolution": [320, 240],
        "game_state": {"name": "combate"},
        "player_health_bar_candidate": {
            "center": [100, 100],
            "x": 80,
            "y": 95,
            "width": 40,
            "height": 10,
            "confidence": 0.9,
        },
    }


class DatasetTests(unittest.TestCase):
    def test_crop_native_pixels_and_reject_aspect_change(self):
        original = np.random.default_rng(8).integers(
            0, 256, (800, 1200, 3), dtype=np.uint8
        )
        roi, geometry = crop_frame(original, (0.1, 0.125, 0.8, 0.75), 1.5)
        np.testing.assert_array_equal(roi, original[100:700, 120:1080])
        self.assertEqual(geometry["roi_xywh"], [120, 100, 960, 600])
        roi[:] = 0
        self.assertTrue(np.any(original[100:700, 120:1080]))
        with self.assertRaisesRegex(ValueError, "Proporção"):
            crop_frame(np.zeros((700, 1200, 3), np.uint8), (0.1, 0.1, 0.8, 0.8), 1.5)
        with self.assertRaises(ValueError):
            crop_frame(original, (0, 0, 0.1, 0.1))

    def test_quality_uses_saved_roi_not_bright_pixels_outside_crop(self):
        raw = np.full((400, 600, 3), 200, np.uint8)
        raw[50:350, 100:500] = 0
        crop, geometry = crop_frame(raw, (1 / 6, 0.125, 2 / 3, 0.75))
        self.assertFalse(QualityMonitor().inspect(crop, 1)["usable"])
        self.assertEqual(geometry["roi_xywh"], [100, 50, 400, 300])

    def test_roi_preferences_roundtrip_and_legacy_preferences(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = SessionOptions(
                roi=(0.1, 0.1, 0.8, 0.8), source_aspect=1.5, burst_fps=5
            )
            save_preferences(tmp, options)
            self.assertEqual(load_preferences(tmp), options)
            write_json(Path(tmp) / "preferencias.json", {"record_video": False})
            self.assertFalse(load_preferences(tmp).record_video)
            self.assertEqual(load_preferences(tmp).burst_fps, 10)
        for roi in ((0, 0, 2, 1), (0, 0, float("nan"), 1)):
            with self.assertRaises(ValueError):
                SessionOptions(roi=roi)

    def test_quality_black_is_excluded_static_scene_only_flagged(self):
        monitor = QualityMonitor()
        black = np.zeros((240, 320, 3), np.uint8)
        self.assertFalse(monitor.inspect(black, 1)["usable"])
        gray = black + 80
        self.assertTrue(monitor.inspect(gray, 2)["usable"])
        monitor.inspect(gray, 2.1)
        result = monitor.inspect(gray, 4.2)
        self.assertIn("possible_freeze_or_static_scene", result["flags"])
        self.assertTrue(result["usable"])
        changed = monitor.inspect(gray + 100, 4.3)
        self.assertNotIn("possible_freeze_or_static_scene", changed["flags"])
        self.assertIn("possible_flash_or_scene_change", changed["flags"])
        self.assertIn(
            "source_resized",
            monitor.inspect(np.full((480, 640, 3), 80, np.uint8), 5)["flags"],
        )

    def test_window_fails_instead_of_silently_cropping_offscreen(self):
        source = WindowCapture.__new__(WindowCapture)
        source.handle = 7
        source._mss = Mock(
            monitors=[{"left": 0, "top": 0, "width": 1920, "height": 1080}]
        )
        with patch(
            "brawlla.capture._window_client_bbox", return_value=(-10, 0, 1280, 720)
        ):
            with self.assertRaisesRegex(RuntimeError, "parcialmente"):
                source.grab()
        source._mss.grab.assert_not_called()

    def test_png_lossless_and_exact_frame_analysis_link(self):
        raw = np.random.default_rng(8).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            rec = SampleRecorder(Path(tmp), session_id="match-one")
            rec.start()
            rec.observe(packet(42, 20, raw, roi_xywh=[20, 10, 640, 480]), data())
            rec.stop()
            rec.join()
            self.assertEqual(rec.error, "")
            entry = load_entries(rec.folder)[0]
            np.testing.assert_array_equal(
                cv2.imread(str(rec.folder / entry["image"])), raw
            )
            pred = json.loads((rec.folder / entry["prediction"]).read_text())
            self.assertEqual(
                (entry["session_id"], entry["frame_id"]), ("match-one", 42)
            )
            self.assertEqual(entry["capture_timestamp_s"], 20)
            self.assertEqual(entry["captured_unix"], 1700000020)
            self.assertEqual(pred["proposed_boxes"][0]["bbox"], [160, 190, 80, 20])
            self.assertFalse(pred["ground_truth"])
            self.assertFalse((rec.folder / "labels").exists())

    def test_sampling_no_duplicate_id_queue_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = SampleRecorder(Path(tmp), sample_fps=15)
            for seq in range(1, 10):
                rec.observe(packet(seq, seq), {})
            rec.observe(packet(9, 20), {})
            self.assertEqual(len(rec._queue), 3)
            self.assertEqual(rec.skipped, 6)
            self.assertEqual([q[0].sequence for q in rec._queue], [7, 8, 9])

    def test_adaptive_burst_starts_immediately_expires_and_cools_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = SampleRecorder(Path(tmp))
            rec.observe(packet(1, 1), {})
            difficult = {"game_state": {"name": "combate", "uncertain": True}}
            rec.observe(packet(2, 1.1), difficult)
            self.assertAlmostEqual(rec._next_sample_at, 1.2)
            self.assertIn("uncertain_hud", rec._queue[-1][2])
            rec.observe(packet(3, 4.2), difficult)
            self.assertAlmostEqual(rec._next_sample_at, 4.7)
            self.assertEqual(rec._queue[-1][2], ["periodic"])
            rec.observe(packet(4, 11.2), difficult)
            self.assertAlmostEqual(rec._next_sample_at, 11.3)

    def test_burst_for_low_health_overlap_and_player_jump(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = SampleRecorder(Path(tmp), burst_fps=5)
            rec.observe(packet(1, 1), data())
            changed = data()
            changed["player_health_bar_candidate"].update(
                center=[200, 100], bar_color="vermelha"
            )
            changed["red_health_bar_candidates"] = [{"center": [201, 101]}]
            rec.observe(packet(2, 1.1), changed)
            self.assertTrue(
                {
                    "possible_low_health",
                    "possible_overlap",
                    "player_position_jump",
                }.issubset(rec._queue[-1][2])
            )
            self.assertAlmostEqual(rec._next_sample_at, 1.3)

    def test_black_samples_and_disk_limits_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = SampleRecorder(Path(tmp))
            rec.start()
            rec.observe(
                packet(1, 1, quality={"usable": False, "flags": ["black_frame"]}), {}
            )
            rec.stop()
            rec.join()
            self.assertEqual(rec.saved, 0)
            self.assertEqual(rec.snapshot()["invalid"], 1)
        for kwargs in ({"max_bytes": 1}, {"min_free_bytes": 10**18}):
            with tempfile.TemporaryDirectory() as tmp:
                rec = SampleRecorder(Path(tmp), **kwargs)
                rec.start()
                rec.observe(packet(1, 1), {})
                rec.stop()
                rec.join()
                self.assertEqual(rec.saved, 0)
                self.assertTrue(rec.limit_reached)
                self.assertFalse(list(rec.folder.glob("*.png")))

    def test_review_negative_requires_explicit_choice_and_valid_boxes(self):
        entry = {
            "width": 320,
            "height": 240,
            "image": "frame_000000001.png",
            "session_id": "game",
            "frame_id": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "negativo"):
                save_annotation(tmp, entry, {"status": "accepted", "boxes": []})
            save_annotation(
                tmp, entry, {"status": "accepted", "boxes": [], "negative": True}
            )
            label = json.loads((Path(tmp) / "labels/frame_000000001.json").read_text())
            self.assertEqual(label["reviewed_by"], "human")
            for box in (
                {"category": "jogador", "bbox": [310, 10, 30, 40]},
                {"category": "bad", "bbox": [10, 10, 20, 20]},
                {"category": "caixa", "bbox": [float("nan"), 10, 20, 20]},
            ):
                with self.assertRaises(ValueError):
                    save_annotation(tmp, entry, {"status": "corrected", "boxes": [box]})

    def test_export_only_human_labels_and_split_is_stable_by_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = SampleRecorder(root, session_id="one-match")
            rec.start()
            for i in range(1, 4):
                rec.observe(packet(i, float(i)), {})
            rec.stop()
            rec.join()
            entries = load_entries(rec.folder)
            self.assertEqual(len(entries), 3)
            save_annotation(
                rec.folder,
                entries[0],
                {"status": "accepted", "boxes": [], "negative": True},
            )
            box = {"category": "jogador", "bbox": [20, 30, 50, 60]}
            save_annotation(
                rec.folder, entries[1], {"status": "corrected", "boxes": [box]}
            )
            save_annotation(
                rec.folder, entries[2], {"status": "uncertain", "boxes": [box]}
            )
            result = export_dataset(root, root / "export")
            self.assertEqual(result["images"], 2)
            self.assertEqual(result["annotations"], 1)
            split = result["sessions"]["one-match"]
            coco = json.loads((root / "export" / (split + ".json")).read_text())
            self.assertEqual(len(coco["images"]), 2)
            self.assertEqual(coco["annotations"][0]["bbox"], [20, 30, 50, 60])
            save_annotation(
                rec.folder, entries[2], {"status": "accepted", "boxes": [box]}
            )
            again = export_dataset(root, root / "export")
            self.assertEqual(result["sessions"], again["sessions"])
            self.assertEqual(again["images"], 3)
            labelpath = (
                rec.folder / "labels" / (Path(entries[2]["image"]).stem + ".json")
            )
            value = json.loads(labelpath.read_text())
            value["frame_id"] = 999
            write_json(labelpath, value)
            with self.assertRaisesRegex(ValueError, "outro quadro"):
                export_dataset(root, root / "export")
            value["frame_id"] = entries[2]["frame_id"]
            write_json(labelpath, value)
            imagepath = rec.folder / entries[2]["image"]
            cv2.imwrite(str(imagepath), np.full((120, 160, 3), 80, np.uint8))
            with self.assertRaisesRegex(ValueError, "dimensão incompatível"):
                export_dataset(root, root / "export")
            imagepath.unlink()
            with self.assertRaisesRegex(ValueError, "Imagem ausente"):
                export_dataset(root, root / "export")

    def test_old_jpeg_sessions_are_preserved_and_skipped_on_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "old"
            folder.mkdir()
            (folder / "manifest.jsonl").write_text(
                json.dumps({"image": "frame_000001.jpg", "width": 320, "height": 240})
                + "\n"
            )
            result = export_dataset(Path(tmp), Path(tmp) / "export")
            self.assertEqual(result["legacy_sessions_skipped"], ["old"])
            with self.assertRaisesRegex(ValueError, "antigas"):
                load_entries(folder)

    def test_manifest_cannot_escape_selected_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = {
                "session_id": "a",
                "frame_id": 1,
                "prediction": "a.prediction.json",
                "image": "../escape.png",
                "width": 320,
                "height": 240,
            }
            (Path(tmp) / "manifest.jsonl").write_text(json.dumps(entry))
            with self.assertRaisesRegex(ValueError, "Caminho"):
                load_entries(tmp)

    def test_pipeline_delivers_raw_packet_and_scaled_analysis_without_overlay(self):
        raw = np.full((480, 800, 3), 70, np.uint8)
        done = threading.Event()
        observations = []
        results = []

        class Source:
            quality_flags = ["possible_window_occlusion"]

            def grab(self):
                return raw

            def close(self):
                pass

        def analyzer(frame, timestamp_s):
            annotated = frame.copy()
            annotated[:] = 255
            return annotated, {"resolution": [frame.shape[1], frame.shape[0]]}

        def observe(packet, data):
            observations.append((packet, data))
            done.set()

        pipe = VisionPipeline(
            Source,
            analyzer,
            lambda im, d: results.append(d),
            max_width=320,
            timestamped_analyzer=True,
            on_observation=observe,
            session_id="sync",
            roi=(0.1, 0.1, 0.8, 0.8),
            source_aspect=800 / 480,
        )
        pipe.start()
        try:
            self.assertTrue(done.wait(3))
        finally:
            pipe.stop()
            pipe.join()
        packet_, result = observations[0]
        np.testing.assert_array_equal(
            packet_.image, np.full((384, 640, 3), 70, np.uint8)
        )
        self.assertEqual(result["frame_id"], packet_.sequence)
        self.assertEqual(result["capture_timestamp_s"], packet_.timestamp_s)
        self.assertEqual(result["session_id"], "sync")
        self.assertIn(
            "possible_window_occlusion", result["capture"]["quality"]["flags"]
        )
        self.assertEqual(result["capture"]["raw_per_analysis_xy"], [2, 2])
        self.assertEqual(pipe.snapshot()["error"], "")

    def test_video_repeats_map_to_source_and_not_training_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = AnalysisRecorder(Path(tmp), fps=5)
            rec.start()
            for seq, now in ((10, 10), (20, 11)):
                rec.submit(
                    np.full((240, 320, 3), 80, np.uint8),
                    {
                        "session_id": "sync",
                        "frame_id": seq,
                        "capture_timestamp_s": now,
                        "capture": {"captured_unix": 1700000000 + now},
                    },
                )
                # Aguarda a gravação para não testar o descarte intencional da fila.
                import time

                deadline = time.monotonic() + 3
                while (
                    rec.records < (1 if seq == 10 else 2)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
            rec.stop()
            self.assertTrue(rec.join(5))
            events = [
                json.loads(line) for line in rec.log_path.read_text().splitlines()
            ]
            self.assertEqual(
                events[1]["video_repeats"],
                {"start": 1, "count": 4, "source_frame_id": 10},
            )
            self.assertEqual(events[1]["frame_id"], 20)
            self.assertEqual(events[1]["capture_timestamp_s"], 11)
            self.assertEqual(rec.written, 6)

    def test_video_preserves_aspect_when_source_changes_shape(self):
        image = np.full((100, 400, 3), 80, np.uint8)
        frame, rect = fit_video_frame(image, 320, 240)
        self.assertEqual(rect, [0, 80, 320, 80])
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(int(frame[:80].max()), 0)

    def test_controller_same_session_id_manifest_audit_and_events(self):
        ready = threading.Event()

        class Source:
            def grab(self):
                return np.full((240, 320, 3), 80, np.uint8)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            c = SessionController(
                lambda kind, payload: ready.set() if kind == "frame" else None, root=tmp
            )
            c.start(Source, SessionOptions(record_video=True, record_samples=True))
            try:
                self.assertTrue(ready.wait(3))
            finally:
                c.stop()
                self.assertTrue(c.join(5))
            audit = json.loads((c.folder / "sessao.json").read_text())
            entry = load_entries(Path(audit["samples"]["folder"]))[0]
            event = json.loads((c.folder / "analise.jsonl").read_text().splitlines()[0])
            self.assertEqual(entry["session_id"], audit["session_id"])
            self.assertEqual(entry["frame_id"], event["frame_id"])
            self.assertEqual(entry["capture_timestamp_s"], event["capture_timestamp_s"])
            self.assertIsNone(audit["error"])


if __name__ == "__main__":
    unittest.main()
