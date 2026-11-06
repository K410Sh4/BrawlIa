import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from brawlla.config import (
    SessionOptions,
    load_config,
)
from brawlla.domain import FramePacket, Timeline
from brawlla.state import StateStabilizer, StateDetector
from brawlla.tracking import TargetTracker
from brawlla.engine import SessionAnalyzer
from brawlla.metrics import SessionMetrics
from brawlla.session import SessionController
from brawlla.stream import VisionPipeline
from brawlla.reporting import save_report


def obj(x=200, kind="jogador_candidato"):
    return {"center": [x, 200], "kind": kind}


def frame():
    return np.full((720, 1280, 3), (100, 55, 105), np.uint8)


def paste(target, sample, x, y):
    h, w = sample.shape[:2]
    target[y : y + h, x : x + w] = sample


class TemporalStateTests(unittest.TestCase):
    def test_short_hud_loss_held_with_explicit_uncertainty_at_different_fps(self):
        for fps in (5, 15, 30):
            s = StateStabilizer()
            s.update("combate", 0.9, 0)
            for i in range(1, int(fps * 2.4) + 1):
                result = s.update("desconhecido", 0.3, i / fps)
                self.assertEqual(result["name"], "combate")
                self.assertTrue(result["uncertain"])
                self.assertEqual(result["reason"], "hud_temporariamente_ausente")
            self.assertEqual(s.update("desconhecido", 0.3, 2.6)["name"], "desconhecido")

    def test_menu_positive_requires_duration_at_varied_fps(self):
        for fps in (5, 15, 30):
            s = StateStabilizer()
            s.update("combate", 0.9, 0)
            s.update("pre_partida", 0.9, 1)
            for i in range(1, int(fps * 0.39) + 1):
                self.assertEqual(
                    s.update("pre_partida", 0.9, 1 + i / fps)["name"], "combate"
                )
            r = s.update("pre_partida", 0.9, 1.41)
            self.assertEqual(r["name"], "pre_partida")
            self.assertEqual(r["transition"]["reason"], "menu_confirmado")

    def test_terminal_screens_preempt_hud_grace(self):
        for terminal in ("derrota", "resultados"):
            s = StateStabilizer()
            s.update("combate", 0.9, 0)
            s.update("desconhecido", 0.3, 0.1)
            self.assertEqual(s.update(terminal, 0.9, 0.2)["name"], terminal)

    def test_no_hud_cannot_become_preparation(self):
        r = StateDetector().detect(frame(), timestamp_s=0)
        self.assertEqual(r["name"], "desconhecido")

    def test_positive_menu_template(self):
        s = StateDetector()
        f = frame()
        paste(f, s.menu, 1040, 620)
        self.assertEqual(s.detect(f, timestamp_s=0)["name"], "pre_partida")

    def test_cached_hud_does_not_refresh_last_real_evidence(self):
        s = StateStabilizer()
        s.update("combate", 0.9, 0)
        s.update("combate", 0.9, 1, evidence_checked=False)
        self.assertEqual(s.update("desconhecido", 0.3, 2.6)["name"], "desconhecido")

    def test_clock_rejects_backwards_and_nonfinite_time(self):
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Timeline().resolve(bad)
        t = Timeline()
        t.resolve(2)
        with self.assertRaises(ValueError):
            t.resolve(1)


class TemporalTrackingTests(unittest.TestCase):
    def test_same_motion_keeps_id_and_velocity_across_fps(self):
        for fps in (5, 15, 30):
            tracker = TargetTracker()
            first = None
            for i in range(fps + 1):
                r = tracker.update([obj(200 + 60 * i / fps)], timestamp_s=i / fps)[0]
                if first is None:
                    first = r["track_id"]
                self.assertEqual(r["track_id"], first)
            self.assertAlmostEqual(r["screen_motion_px_per_s"][0], 60, places=1)

    def test_occlusion_duration_has_same_effect_at_different_fps(self):
        for fps in (5, 15, 30):
            t = TargetTracker()
            first = t.update([obj()], timestamp_s=0)[0]["track_id"]
            for i in range(1, int(fps * 0.6) + 1):
                t.update([], timestamp_s=i / fps)
            self.assertEqual(t.update([obj()], timestamp_s=0.7)[0]["track_id"], first)
            new = t.update([obj()], timestamp_s=1.6)[0]["track_id"]
            self.assertNotEqual(new, first)
            self.assertTrue(
                any(
                    e["event"] == "lost"
                    and e["reason"] == "tempo_sem_observacao_excedido"
                    for e in t.drain_events()
                )
            )

    def test_expiry_happens_before_new_observation_after_long_gap(self):
        t = TargetTracker()
        a = t.update([obj()], timestamp_s=0)[0]
        b = t.update([obj()], timestamp_s=10)[0]
        self.assertNotEqual(a["track_id"], b["track_id"])

    def test_reset_records_reason_and_does_not_reuse_id(self):
        t = TargetTracker()
        a = t.update([obj()], timestamp_s=0)[0]
        t.reset("resultados_confirmado", 0.1)
        b = t.update([obj()], timestamp_s=0.2)[0]
        self.assertNotEqual(a["track_id"], b["track_id"])
        self.assertTrue(
            any(
                e["event"] == "reset" and e["reason"] == "resultados_confirmado"
                for e in t.drain_events()
            )
        )
        self.assertEqual(t.drain_events(), [])

    def test_player_is_not_associated_with_red_object(self):
        t = TargetTracker()
        a = t.update([obj()], timestamp_s=0)[0]
        b = t.update([obj(kind="indefinido")], timestamp_s=0.1)[0]
        self.assertNotEqual(a["track_id"], b["track_id"])

    def test_nonfinite_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            TargetTracker().update([obj(float("nan"))], timestamp_s=0)


class ArchitectureTests(unittest.TestCase):
    def test_observe_is_pure_and_renderer_preserves_input(self):
        e = SessionAnalyzer()
        f = frame()
        paste(f, e.states.hud, 40, 20)
        name_width = cv2.getTextSize("MEU PLAYER", cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[
            0
        ][0]
        cv2.putText(
            f,
            "MEU PLAYER",
            (638 - name_width // 2, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (30, 230, 50),
            2,
        )
        cv2.rectangle(f, (600, 286), (675, 294), (30, 220, 70), -1)
        cv2.putText(
            f, "7560", (610, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        original = f.copy()
        result, scene = e.observe(f, 0)
        np.testing.assert_array_equal(f, original)
        self.assertEqual(result.to_dict()["schema_version"], 2)
        self.assertTrue(result.tracked_objects)
        image, data = e.analyze(f, 0.2)
        np.testing.assert_array_equal(f, original)
        self.assertFalse(np.array_equal(image, f))
        self.assertIn("track_events", data)

    def test_short_unknown_does_not_reset_player_id(self):
        e = SessionAnalyzer()
        combat = frame()
        paste(combat, e.states.hud, 40, 20)
        name_width = cv2.getTextSize("MEU PLAYER", cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[
            0
        ][0]
        cv2.putText(
            combat,
            "MEU PLAYER",
            (638 - name_width // 2, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (30, 230, 50),
            2,
        )
        cv2.rectangle(combat, (600, 286), (675, 294), (30, 220, 70), -1)
        cv2.putText(
            combat,
            "7560",
            (610, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        _, a = e.analyze(combat, 0)
        first = a["tracked_objects"][0]["track_id"]
        # Remove somente o HUD, mantendo a barra observável durante a perda.
        hidden = combat.copy()
        hidden[:90] = frame()[:90]
        for timestamp in (0.2, 0.4, 0.8, 1.2):
            _, data = e.analyze(hidden, timestamp)
            self.assertEqual(data["game_state"]["name"], "combate")
            self.assertEqual(data["tracked_objects"][0]["track_id"], first)
            self.assertFalse(any(ev["event"] == "reset" for ev in data["track_events"]))

    def test_pipeline_passes_capture_time_to_analyzer(self):
        done = threading.Event()
        times = []

        class Source:
            def grab(self):
                return np.zeros((240, 320, 3), np.uint8)

            def close(self):
                pass

        def analyzer(f, *, timestamp_s):
            times.append(timestamp_s)
            return f, {}

        p = VisionPipeline(
            Source,
            analyzer,
            lambda f, d: done.set(),
            timestamped_analyzer=True,
            target_fps=10,
        )
        p.start()
        self.assertTrue(done.wait(2))
        p.stop()
        p.join()
        self.assertGreater(times[0], 0)

    def test_config_validation_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "config.json"
            for raw in (
                {"state": {"combat_grace_s": -1}},
                {"tracking": {"max_absence_s": float("nan")}},
                {"vision": {"red_lo": [999, 0, 0]}},
                {"state": {"typo": 1}},
                {"unknown": {}},
            ):
                p.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    load_config(p)
            p.write_text(json.dumps({"state": {"combat_grace_s": 1.5}}))
            self.assertEqual(load_config(p).state.combat_grace_s, 1.5)

    def test_bad_session_options_fail_before_resources(self):
        with self.assertRaises(ValueError):
            SessionOptions(capture_fps=99)
        with self.assertRaises(ValueError):
            SessionOptions(max_width=1)

    def test_frame_packet_validates_time(self):
        with self.assertRaises(ValueError):
            FramePacket(1, float("nan"), np.array([1]))


class SessionLifecycleTests(unittest.TestCase):
    def test_start_stop_restart_without_tk_and_finalized_files(self):
        events = []
        ready = threading.Event()

        class Source:
            closed = 0

            def grab(self):
                return np.zeros((240, 320, 3), np.uint8)

            def close(self):
                Source.closed += 1

        def notify(kind, payload):
            events.append((kind, payload))
            if kind == "frame":
                ready.set()

        with tempfile.TemporaryDirectory() as temp:
            c = SessionController(notify, root=Path(temp))
            for _ in range(2):
                ready.clear()
                c.start(Source, SessionOptions(record_video=True, capture_fps=10))
                self.assertTrue(ready.wait(3))
                with self.assertRaises(RuntimeError):
                    c.start(Source, SessionOptions())
                self.assertTrue(c.stop())
                self.assertFalse(c.stop())
                self.assertTrue(c.join(5))
                self.assertFalse(c.active)
            self.assertEqual(Source.closed, 2)
            final = [payload for kind, payload in events if kind == "finalized"]
            self.assertEqual(len(final), 2)
            for result, error in final:
                self.assertIsNone(result["error"])
                self.assertFalse(error)
                self.assertTrue((Path(result["folder"]) / "resumo.json").exists())

    def test_failed_start_finalizes_already_started_sample_recorder(self):
        events = []
        with tempfile.TemporaryDirectory() as temp:
            c = SessionController(lambda *event: events.append(event), root=Path(temp))
            with patch(
                "brawlla.session.AnalysisRecorder.start",
                side_effect=OSError("disk failure"),
            ):
                with self.assertRaises(OSError):
                    c.start(
                        lambda: None,
                        SessionOptions(record_video=True, record_samples=True),
                    )
            self.assertTrue(c.join(3))
            self.assertEqual(c.state, "idle")
            self.assertEqual([kind for kind, _ in events], ["finalized"])
            self.assertIn("disk failure", events[0][1][0]["error"])

    def test_capture_error_is_preserved_on_finalization(self):
        events = []

        def source():
            raise RuntimeError("source unavailable")

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        c = SessionController(lambda *event: events.append(event), root=Path(temp.name))
        c.start(source, SessionOptions())
        deadline = time.monotonic() + 2
        while not c.pipeline.snapshot()["error"] and time.monotonic() < deadline:
            time.sleep(0.01)
        c.stop()
        self.assertTrue(c.join(3))
        self.assertIn("source unavailable", events[-1][1][0]["error"])


class ResourceBoundaryTests(unittest.TestCase):
    def test_invalid_frame_publish_does_not_advance_sequence(self):
        from brawlla.stream import LatestFrame

        slot = LatestFrame()
        with self.assertRaises(ValueError):
            slot.publish(np.array([1]), captured_at=float("nan"))
        self.assertEqual(slot.publish(np.array([2]), captured_at=1), 1)
        self.assertEqual(slot.take_latest_after(0, 0).sequence, 1)

    def test_unconsumed_tracker_events_are_bounded_and_report_omission(self):
        t = TargetTracker()
        for i in range(300):
            t.update([obj()], timestamp_s=i)
        events = t.drain_events()
        self.assertLessEqual(len(events), 501)
        self.assertTrue(any(e["event"] == "omitted" and e["count"] > 0 for e in events))

    def test_report_failure_still_completes_video_finalization(self):
        from brawlla.recording import AnalysisRecorder

        with tempfile.TemporaryDirectory() as temp:
            recorder = AnalysisRecorder(Path(temp))
            with patch(
                "brawlla.recording.save_report",
                side_effect=RuntimeError("report failure"),
            ):
                recorder.start()
                recorder.stop()
                self.assertTrue(recorder.join(3))
            self.assertIn("report failure", recorder.snapshot()["error"])


class DiagnosticTests(unittest.TestCase):
    def test_inferred_window_counts_suspension_and_uncertainty(self):
        m = SessionMetrics()
        for time_s, state, player, uncertain in [
            (0, "combate", True, False),
            (1, "combate", True, True),
            (2, "desconhecido", False, True),
            (3, "combate", True, False),
            (4, "derrota", False, False),
        ]:
            m.observe(
                time_s,
                {
                    "game_state": {"name": state, "uncertain": uncertain},
                    "player_health_bar_candidate": {"x": 1} if player else None,
                },
            )
        r = m.summary()
        self.assertEqual(r["player_candidate_coverage_pct"], 100)
        self.assertEqual(r["inferred_match_candidate_coverage_pct"], 75)
        self.assertEqual(r["analysis_suspended_in_match_s"], 1)
        self.assertEqual(r["uncertain_combat_frames"], 1)

    def test_legacy_report_remains_supported(self):
        from brawlla.metrics import summarize_log

        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "log.jsonl"
            p.write_text(
                json.dumps(
                    {"elapsed_s": 0, "detections": {"game_state": {"name": "combate"}}}
                )
                + "\n"
            )
            summary = summarize_log(p)
            save_report(summary, Path(temp) / "report.json")
            self.assertIn(
                "Janela",
                (Path(temp) / "report.html")
                .read_text(encoding="utf-8")
                .replace("janela", "Janela"),
            )


if __name__ == "__main__":
    unittest.main()
