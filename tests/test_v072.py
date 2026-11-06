from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from brawlla.config import SessionOptions
from brawlla.storage import load_preferences, save_preferences
from brawlla.session import SessionController
from brawlla.recording import AnalysisRecorder
from brawlla.diagnostics import format_recording_status


class Source:
    def grab(self):
        return np.full((240, 320, 3), 90, np.uint8)

    def close(self):
        pass


class SavingTests(unittest.TestCase):
    def test_ui_preferences_enable_video_by_default_and_remember_explicit_off(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertTrue(load_preferences(root).record_video)
            chosen = SessionOptions(
                record_video=False, record_samples=True, video_fps=30
            )
            save_preferences(root, chosen)
            self.assertEqual(load_preferences(root), chosen)
            save_preferences(root, replace(chosen, record_video=True))
            self.assertTrue(load_preferences(root).record_video)

    def test_recording_writes_replayable_video_and_audit_with_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = []
            ready = threading.Event()

            def notify(kind, data):
                outputs.append((kind, data))
                if kind == "frame":
                    ready.set()

            c = SessionController(notify, root=Path(temp))
            c.start(Source, SessionOptions(record_video=True, record_samples=True))
            try:
                self.assertTrue((c.folder / "sessao.json").exists())
                self.assertTrue(ready.wait(3))
            finally:
                c.stop()
                self.assertTrue(c.join(5))
            audit = json.loads((c.folder / "sessao.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["video_requested"])
            self.assertGreater(audit["video"]["written"], 0)
            self.assertTrue(audit["video"]["verified"])
            self.assertGreater(audit["samples"]["saved"], 0)
            self.assertEqual(audit["status"], "finalizada")
            self.assertIsNone(audit["error"])
            report = json.loads(
                (c.folder / "relatorio.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["recording"]["images_saved"], audit["samples"]["saved"]
            )
            self.assertTrue(report["recording"]["video_verified"])
            self.assertIn(
                "Arquivos salvos",
                (c.folder / "relatorio.html").read_text(encoding="utf-8"),
            )
            self.assertTrue(Path(audit["video"]["video"]).is_absolute())
            video = cv2.VideoCapture(audit["video"]["video"])
            try:
                self.assertTrue(video.read()[0])
            finally:
                video.release()
            self.assertIn("verificado", format_recording_status(c.recording_snapshot()))

    def test_session_without_video_records_that_it_was_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            c = SessionController(lambda *e: None, root=Path(temp))
            c.start(Source, SessionOptions(record_video=False))
            folder = c.folder
            c.stop()
            self.assertTrue(c.join(3))
            audit = json.loads((folder / "sessao.json").read_text(encoding="utf-8"))
            self.assertFalse(audit["video_requested"])
            self.assertIsNone(audit["video"])
            self.assertFalse(list(folder.glob("*.mp4")))
            self.assertIn("desligado", format_recording_status(c.recording_snapshot()))

    def test_capture_never_producing_a_frame_does_not_claim_video_was_saved(self):
        class FailedSource:
            def grab(self):
                raise RuntimeError("capture failed")

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            c = SessionController(lambda *e: None, root=Path(temp))
            c.start(FailedSource, SessionOptions(record_video=True))
            deadline = time.monotonic() + 2
            while not c.pipeline.snapshot()["error"] and time.monotonic() < deadline:
                time.sleep(0.01)
            c.stop()
            self.assertTrue(c.join(5))
            audit = json.loads((c.folder / "sessao.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["video"]["written"], 0)
            self.assertFalse(audit["video"]["verified"])
            self.assertIn("capture failed", audit["error"])
            self.assertIn("falha", format_recording_status(c.recording_snapshot()))

    def test_unwritable_destination_prevents_capture_from_starting(self):
        with tempfile.TemporaryDirectory() as temp:
            factory = unittest.mock.Mock(return_value=Source())
            c = SessionController(lambda *e: None, root=Path(temp))
            with patch(
                "brawlla.session.probe_directory",
                side_effect=PermissionError("access denied"),
            ):
                with self.assertRaises(PermissionError):
                    c.start(factory, SessionOptions(record_video=True))
            self.assertTrue(c.join(3))
            factory.assert_not_called()
            self.assertIn("access denied", c.recording_error)

    def test_codec_failure_is_reported_in_summary_and_session_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            c = SessionController(lambda *e: None, root=Path(temp))
            with patch(
                "brawlla.recording.AnalysisRecorder._open_writer",
                side_effect=RuntimeError("codec unavailable"),
            ):
                c.start(Source, SessionOptions(record_video=True))
                deadline = time.monotonic() + 3
                while not c.video.snapshot()["done"] and time.monotonic() < deadline:
                    time.sleep(0.01)
                c.stop()
                self.assertTrue(c.join(5))
            audit = json.loads((c.folder / "sessao.json").read_text(encoding="utf-8"))
            summary = json.loads((c.folder / "resumo.json").read_text(encoding="utf-8"))
            self.assertIn("codec unavailable", audit["error"])
            self.assertIn("codec unavailable", summary["error"])
            self.assertFalse(summary["video_verified"])

    def test_report_failure_does_not_lose_summary_file(self):
        with tempfile.TemporaryDirectory() as temp:
            r = AnalysisRecorder(Path(temp))
            with patch(
                "brawlla.recording.save_report",
                side_effect=RuntimeError("report unavailable"),
            ):
                r.start()
                r.stop()
                self.assertTrue(r.join(3))
            summary = json.loads(r.summary_path.read_text(encoding="utf-8"))
            self.assertIn("report unavailable", summary["error"])
            self.assertIsNone(summary["report_file"])

    def test_destination_does_not_follow_later_cwd_changes(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "store"
            other = Path(temp) / "other"
            other.mkdir()
            c = SessionController(lambda *e: None, root=root)
            try:
                os.chdir(other)
                c.start(Source, SessionOptions(record_video=False))
                c.stop()
                self.assertTrue(c.join(3))
                self.assertTrue(c.folder.is_relative_to(root))
                self.assertTrue((c.folder / "sessao.json").exists())
                self.assertFalse((other / "gravacoes").exists())
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
