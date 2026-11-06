import threading
import unittest
from unittest.mock import patch
import numpy as np
from brawlla.stream import VisionPipeline
from brawlla.engine import SessionAnalyzer
from brawlla.diagnostics import (
    format_frame_counts,
    format_processing_health,
    target_status,
)


class DiagnosticsTests(unittest.TestCase):
    def test_frame_counts_show_processed_and_replaced_separately(self):
        text = format_frame_counts(
            {"captured": 30, "analyzed": 15, "replaced_before_analysis": 14}
        )
        self.assertIn("capturados: 30", text)
        self.assertIn("Analisados: 15", text)
        self.assertIn("Substituídos antes da análise: 14", text)
        self.assertNotIn("ignorados", text)

    def test_noncombat_state_is_not_an_unprocessed_frame(self):
        e = SessionAnalyzer()
        _, data = e.analyze(np.zeros((240, 320, 3), np.uint8), timestamp_s=0)
        self.assertFalse(data["target_analysis"]["enabled"])
        self.assertIn("HUD não reconhecido", data["target_analysis"]["message"])
        self.assertIn("total", data["timings_ms"])

    def test_uncertain_combat_still_has_target_analysis_enabled(self):
        r = target_status({"name": "combate", "uncertain": True})
        self.assertTrue(r["enabled"])
        self.assertIn("incerto", r["message"])

    def test_30fps_capture_15fps_analysis_produces_results_despite_replacements(self):
        outputs = []
        ready = threading.Event()

        class Source:
            def grab(self):
                return np.zeros((240, 320, 3), np.uint8)

            def close(self):
                pass

        analyzer = SessionAnalyzer()

        def result(frame, data):
            outputs.append(data)
            if len(outputs) >= 6:
                ready.set()

        p = VisionPipeline(
            Source,
            analyzer.analyze,
            result,
            target_fps=30,
            analysis_fps=15,
            timestamped_analyzer=True,
        )
        p.start()
        try:
            self.assertTrue(ready.wait(3))
        finally:
            p.stop()
            p.join()
        s = p.snapshot()
        self.assertGreaterEqual(s["analyzed"], 6)
        self.assertGreater(s["captured"], s["analyzed"])
        self.assertGreater(s["replaced_before_analysis"], 0)
        self.assertEqual(s["skipped"], s["replaced_before_analysis"])
        self.assertFalse(s["error"])
        for d in outputs:
            self.assertFalse(d["target_analysis"]["enabled"])
            self.assertGreaterEqual(
                d["processing_counters"]["captured"],
                d["processing_counters"]["analyzed"],
            )

    def test_stalled_analysis_with_fresh_capture_is_detected(self):
        p = VisionPipeline(None, None, None)
        p._started_at = 0
        p._last_capture_at = 2.8
        p._last_result_at = 0
        with patch("brawlla.stream.time.perf_counter", return_value=3):
            s = p.snapshot()
        self.assertTrue(s["analysis_stalled"])
        self.assertFalse(s["capture_stalled"])
        self.assertIn("análise está sem resultados", format_processing_health(s))

    def test_stalled_capture_is_distinguished_from_analysis_stall(self):
        p = VisionPipeline(None, None, None)
        p._started_at = 0
        with patch("brawlla.stream.time.perf_counter", return_value=3):
            s = p.snapshot()
        self.assertTrue(s["capture_stalled"])
        self.assertFalse(s["analysis_stalled"])
        self.assertIn("captura", format_processing_health(s))

    def test_stopped_pipeline_does_not_report_a_stall(self):
        p = VisionPipeline(None, None, None)
        p._started_at = 0
        p.stop()
        with patch("brawlla.stream.time.perf_counter", return_value=3):
            s = p.snapshot()
        self.assertFalse(s["capture_stalled"])
        self.assertFalse(s["analysis_stalled"])

    def test_errors_take_priority_in_diagnostic(self):
        self.assertIn(
            "source error",
            format_processing_health(
                {"error": "source error", "capture_stalled": True}
            ),
        )


if __name__ == "__main__":
    unittest.main()
