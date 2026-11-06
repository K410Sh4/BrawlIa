import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from brawlla.engine import SessionAnalyzer
from brawlla.state import StateDetector, StateStabilizer
from brawlla.tracking import TargetTracker
from brawlla.metrics import SessionMetrics, summarize_log, save_report
from brawlla.ui_mailbox import UiMailbox
from brawlla.stream import LatestFrame, VisionPipeline
from brawlla.recording import AnalysisRecorder


def obj(x, kind="personagem_possivel"):
    return {"kind": kind, "center": [x, 100]}


class StateRegressionTests(unittest.TestCase):
    def test_candidates_need_consecutive_time_not_accumulated_votes(self):
        s = StateStabilizer()
        s.update("combate", 0.9, 0)
        s.update("pre_partida", 0.8, 0.1)
        self.assertEqual(s.update("desconhecido", 0.4, 0.2)["name"], "combate")
        self.assertEqual(s.update("pre_partida", 0.8, 0.3)["pending_votes"], 1)
        self.assertEqual(s.update("pre_partida", 0.8, 0.55)["name"], "combate")
        self.assertEqual(s.update("pre_partida", 0.8, 0.71)["name"], "pre_partida")

    def test_unknown_has_a_bounded_time_grace(self):
        s = StateStabilizer()
        s.update("combate", 0.9, 0)
        for t in (0.1, 0.5, 2.0):
            result = s.update("desconhecido", 0.45, t)
            self.assertEqual(result["name"], "combate")
            self.assertTrue(result["uncertain"])
        self.assertEqual(s.update("desconhecido", 0.45, 2.51)["name"], "desconhecido")

    def test_defeat_preempts_pending_state(self):
        s = StateStabilizer()
        s.update("combate", 0.8)
        s.update("pre_partida", 0.7)
        self.assertEqual(s.update("derrota", 0.9)["name"], "derrota")

    def test_stateless_probe_does_not_change_session(self):
        s = StateDetector()
        frame = np.zeros((240, 320, 3), np.uint8)
        s.detect(frame)
        before = (s._calls, s.stabilizer.state, s._last_result.copy())
        s.detect(frame, update=False)
        self.assertEqual(before, (s._calls, s.stabilizer.state, s._last_result))

    def test_cached_result_does_not_repeat_change_event(self):
        s = StateDetector()
        frame = np.zeros((720, 1280, 3), np.uint8)
        h, w = s.hud.shape[:2]
        frame[20 : 20 + h, 40 : 40 + w] = s.hud
        self.assertTrue(s.detect(frame)["changed"])
        cached = s.detect(frame)
        self.assertFalse(cached["changed"])
        self.assertTrue(cached["cached"])

    def test_unknown_transition_does_not_reset_tracks(self):
        engine = SessionAnalyzer()
        track = engine.tracker.update([obj(100)], 320)[0]["track_id"]
        engine.previous_state = "combate"
        with patch.object(
            engine.states,
            "detect",
            return_value={"name": "desconhecido", "confidence": 0.4},
        ):
            engine.analyze(np.zeros((240, 320, 3), np.uint8))
        # Mesma resolução da análise anterior.
        self.assertEqual(track, engine.tracker.update([obj(100)], 320)[0]["track_id"])


class TrackingRegressionTests(unittest.TestCase):
    def test_global_pair_priority_avoids_stealing_closest_track(self):
        t = TargetTracker()
        a, b = t.update([obj(100), obj(160)])
        c, d = t.update([obj(140), obj(158)])
        self.assertEqual((a["track_id"], b["track_id"]), (c["track_id"], d["track_id"]))

    def test_prediction_preserves_crossing_tracks(self):
        t = TargetTracker()
        a, b = t.update([obj(100), obj(200)])
        t.update([obj(140), obj(160)])
        c, d = t.update([obj(120), obj(180)])
        self.assertEqual(c["track_id"], b["track_id"])
        self.assertEqual(d["track_id"], a["track_id"])

    def test_short_occlusion_preserves_id(self):
        t = TargetTracker()
        first = t.update([obj(100)])[0]
        t.update([obj(120)])
        t.update([])
        self.assertEqual(t.update([obj(160)])[0]["track_id"], first["track_id"])

    def test_changed_red_class_preserves_id(self):
        t = TargetTracker()
        first = t.update([obj(100, "caixa_provavel")])[0]
        self.assertEqual(first["track_id"], t.update([obj(105)])[0]["track_id"])

    def test_green_player_cannot_match_undefined_red(self):
        t = TargetTracker()
        first = t.update([obj(100, "jogador_candidato")])[0]
        self.assertNotEqual(
            first["track_id"], t.update([obj(101, "indefinido")])[0]["track_id"]
        )

    def test_expired_track_not_reused(self):
        t = TargetTracker(max_missed=1)
        first = t.update([obj(100)])[0]
        t.update([])
        t.update([])
        self.assertNotEqual(first["track_id"], t.update([obj(100)])[0]["track_id"])

    def test_resolution_change_resets_coordinates(self):
        t = TargetTracker()
        a = t.update([obj(100)], 1280)[0]
        self.assertNotEqual(a["track_id"], t.update([obj(100)], 640)[0]["track_id"])

    def test_order_does_not_change_association(self):
        t = TargetTracker()
        u = TargetTracker()
        for tracker in (t, u):
            tracker.update([obj(100), obj(160)])
        a = t.update([obj(140), obj(158)])
        b = u.update([obj(158), obj(140)])
        self.assertEqual([o["track_id"] for o in a], [o["track_id"] for o in b][::-1])


class MetricsTests(unittest.TestCase):
    def test_coverage_uses_combat_only_and_durations_use_intervals(self):
        s = SessionMetrics()
        s.observe(0, {"game_state": {"name": "pre_partida"}})
        s.observe(
            2,
            {
                "game_state": {"name": "combate"},
                "player_health_bar_candidate": {"x": 1},
            },
        )
        s.observe(3, {"game_state": {"name": "combate"}})
        s.observe(5, {"game_state": {"name": "derrota"}})
        out = s.summary()
        self.assertEqual(out["player_candidate_coverage_pct"], 50)
        self.assertEqual(out["state_duration_s"], {"pre_partida": 2, "combate": 3})
        self.assertEqual(out["state_transitions"], 2)

    def test_empty_report_does_not_invent_coverage(self):
        s = SessionMetrics().summary()
        self.assertIsNone(s["player_candidate_coverage_pct"])
        self.assertEqual(s["observation_duration_s"], 0)

    def test_invalid_timestamps_rejected(self):
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SessionMetrics().observe(bad, {})
        s = SessionMetrics()
        s.observe(2, {})
        with self.assertRaises(ValueError):
            s.observe(1, {})

    def test_timing_window_is_bounded_and_mean_uses_all(self):
        s = SessionMetrics()
        for i in range(2100):
            s.observe(
                i, {"timings_ms": {"total": i}, "pipeline_timings_ms": {"capture": 1}}
            )
        out = s.summary()["timings_ms"]
        self.assertEqual(out["total"]["mean"], 1049.5)
        self.assertEqual(out["total"]["p95_window_samples"], 2048)
        self.assertEqual(out["pipeline.capture"]["mean"], 1)

    def test_html_escapes_untrusted_state_and_creates_no_scripts(self):
        with tempfile.TemporaryDirectory() as temp:
            s = SessionMetrics()
            s.observe(0, {"game_state": {"name": "<script>alert(1)</script>"}})
            p = Path(temp) / "report.json"
            save_report(s.summary(), p)
            html = p.with_suffix(".html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            with self.assertRaises(ValueError):
                save_report(s.summary(), p.with_suffix(".html"))

    def test_legacy_log_and_invalid_line_message(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "log.jsonl"
            p.write_text(
                json.dumps({"elapsed_s": 0, "detections": {}}) + "\n", encoding="utf-8"
            )
            self.assertEqual(summarize_log(p)["analyzed_frames"], 1)
            with p.open("a") as f:
                f.write("{broken\n")
            with self.assertRaisesRegex(ValueError, "linha 2"):
                summarize_log(p)


class PipelineAndUiTests(unittest.TestCase):
    def test_preview_pressure_cannot_drop_finalization_or_single_result(self):
        mailbox = UiMailbox()
        mailbox.offer("single", 1)
        mailbox.offer("finalized", 2)
        for i in range(1000):
            mailbox.offer("frame", i)
        self.assertEqual(
            mailbox.drain(), [("frame", 999), ("single", 1), ("finalized", 2)]
        )
        self.assertEqual(mailbox.drain(), [])

    def test_capture_metadata_belongs_to_same_frame(self):
        slot = LatestFrame()
        slot.publish(np.array([1]), 4.0, 100.0)
        slot.publish(np.array([2]), 7.0, 200.0)
        seq, frame, started, ms = slot.take_packet_after(0, 0)
        self.assertEqual((seq, int(frame[0]), started, ms), (2, 2, 200.0, 7.0))

    def test_pipeline_outputs_timing_metadata(self):
        done = __import__("threading").Event()
        out = []

        class Source:
            def grab(self):
                return np.zeros((240, 320, 3), np.uint8)

            def close(self):
                pass

        def result(frame, data):
            out.append(data)
            done.set()

        p = VisionPipeline(Source, lambda f: (f, {}), result, target_fps=10)
        p.start()
        self.assertTrue(done.wait(3))
        p.stop()
        p.join()
        timings = out[0]["pipeline_timings_ms"]
        self.assertGreaterEqual(
            timings["frame_age_at_result"], timings["analysis_with_resize"]
        )
        self.assertIn("analysis_p95_ms", p.snapshot())

    def test_invalid_width_fails_before_workers_start(self):
        with self.assertRaises(ValueError):
            VisionPipeline(None, None, None, max_width=1)

    def test_recording_includes_report_and_phase_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            rec = AnalysisRecorder(Path(temp), fps=5)
            rec.start()
            rec.submit(
                np.zeros((240, 320, 3), np.uint8),
                {"game_state": {"name": "combate"}, "timings_ms": {"total": 2}},
            )
            rec.stop()
            self.assertTrue(rec.join(5))
            self.assertIsNone(rec.error)
            summary = json.loads(rec.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["session_metrics"]["analyzed_frames"], 1)
            self.assertTrue((rec.folder / "relatorio.html").exists())


if __name__ == "__main__":
    unittest.main()
