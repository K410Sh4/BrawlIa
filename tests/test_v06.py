import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np
from brawlla.player import PlayerLocator
from brawlla.vision import analyze
from brawlla.state import StateDetector
from brawlla.engine import SessionAnalyzer


def canvas():
    return np.full((720, 1280, 3), (100, 55, 105), np.uint8)


def player_frame(fill=25, color=(20, 220, 245), with_name=True, with_number=True):
    f = canvas()
    if with_name:
        text = "MEU PLAYER"
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = cv2.getTextSize(text, font, 0.60, 2)[0][0]
        cv2.putText(f, text, (640 - width // 2, 200), font, 0.60, (30, 230, 50), 2)
    cv2.rectangle(f, (600, 226), (680, 239), (30, 25, 35), -1)
    if fill:
        cv2.rectangle(f, (603, 229), (603 + fill - 1, 235), color, -1)
    if with_number:
        cv2.putText(
            f, "385", (622, 223), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 250, 250), 1
        )
    return f


def paste(target, sample, x, y):
    h, w = sample.shape[:2]
    target[y : y + h, x : x + w] = sample


class PlayerTests(unittest.TestCase):
    def test_yellow_bar_under_green_name_is_player_candidate(self):
        _, d = analyze(player_frame())
        p = d["player_health_bar_candidate"]
        self.assertIsNotNone(p)
        self.assertEqual(p["bar_color"], "amarela")
        self.assertLess(abs(p["center"][0] - 640), 10)

    def test_critical_fill_of_two_pixels_uses_observed_frame(self):
        _, d = analyze(player_frame(fill=2))
        p = d["player_health_bar_candidate"]
        self.assertIsNotNone(p)
        self.assertIn(
            p["player_evidence"], ("nome_verde_glifos_e_moldura", "nome_verde_e_barra")
        )
        self.assertTrue(p["container_width_estimated"])

    def test_red_flash_bar_stays_player_when_anchored_to_green_name(self):
        _, d = analyze(player_frame(fill=25, color=(50, 40, 232)))
        p = d["player_health_bar_candidate"]
        self.assertIsNotNone(p)
        self.assertEqual(p["bar_color"], "vermelha")
        self.assertEqual(d["red_health_bar_candidates"], [])

    def test_white_flash_bar_under_name(self):
        _, d = analyze(player_frame(fill=70, color=(245, 245, 245)))
        self.assertIsNotNone(d["player_health_bar_candidate"])

    def test_yellow_bar_without_name_is_not_player(self):
        _, d = analyze(player_frame(with_name=False))
        self.assertIsNone(d["player_health_bar_candidate"])

    def test_name_without_bar_or_number_is_not_enough(self):
        f = player_frame(fill=0, with_number=False)
        _, d = analyze(f)
        self.assertIsNone(d["player_health_bar_candidate"])

    def test_center_does_not_follow_shrinking_fill(self):
        _, a = analyze(player_frame(fill=70))
        _, b = analyze(player_frame(fill=8))
        self.assertLess(
            abs(
                a["player_health_bar_candidate"]["center"][0]
                - b["player_health_bar_candidate"]["center"][0]
            ),
            10,
        )

    def test_previous_location_cannot_invent_observation_on_empty_frame(self):
        locator = PlayerLocator()
        self.assertIsNotNone(
            analyze(player_frame(), locator)[1]["player_health_bar_candidate"]
        )
        self.assertIsNone(analyze(canvas(), locator)[1]["player_health_bar_candidate"])

    def test_green_scenery_without_number_is_not_player(self):
        f = canvas()
        cv2.rectangle(f, (600, 400), (670, 410), (30, 220, 70), -1)
        self.assertIsNone(analyze(f)[1]["player_health_bar_candidate"])

    def test_half_resolution_yellow_player(self):
        f = cv2.resize(player_frame(), (640, 360), interpolation=cv2.INTER_AREA)
        _, d = analyze(f)
        p = d["player_health_bar_candidate"]
        self.assertIsNotNone(p)
        self.assertLess(abs(p["center"][0] - 320), 10)


class BarFilterTests(unittest.TestCase):
    def test_red_rectangle_without_number_is_rejected(self):
        f = canvas()
        cv2.rectangle(f, (300, 300), (370, 309), (50, 40, 232), -1)
        self.assertEqual(analyze(f)[1]["red_health_bar_candidates"], [])

    def test_hp_digits_support_a_red_bar(self):
        f = canvas()
        cv2.rectangle(f, (300, 300), (370, 309), (50, 40, 232), -1)
        cv2.putText(
            f, "3500", (310, 296), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1
        )
        self.assertEqual(len(analyze(f)[1]["red_health_bar_candidates"]), 1)

    def test_pink_scenery_is_not_white_number_evidence(self):
        f = canvas()
        cv2.rectangle(f, (285, 270), (390, 320), (160, 150, 245), -1)
        cv2.rectangle(f, (300, 300), (370, 309), (50, 40, 232), -1)
        self.assertEqual(analyze(f)[1]["red_health_bar_candidates"], [])

    def test_hud_and_footer_candidates_rejected(self):
        f = canvas()
        for y in (12, 704):
            cv2.rectangle(f, (300, y), (370, y + 8), (50, 40, 232), -1)
            cv2.putText(
                f,
                "3500",
                (310, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
        self.assertEqual(analyze(f)[1]["red_health_bar_candidates"], [])

    def test_half_resolution_bar_with_digits(self):
        f = canvas()
        cv2.rectangle(f, (300, 300), (370, 309), (50, 40, 232), -1)
        cv2.putText(
            f, "3500", (310, 296), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        f = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
        self.assertGreaterEqual(len(analyze(f)[1]["red_health_bar_candidates"]), 1)


class ResultsTests(unittest.TestCase):
    def test_results_takes_priority_and_suspends_targets(self):
        engine = SessionAnalyzer()
        f = canvas()
        paste(f, engine.states.results, 24, 12)
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
        cv2.rectangle(f, (600, 286), (672, 294), (30, 220, 70), -1)
        cv2.putText(
            f, "7560", (610, 281), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        _, d = engine.analyze(f)
        self.assertEqual(d["game_state"]["name"], "resultados")
        self.assertEqual(d["tracked_objects"], [])

    def test_half_resolution_results(self):
        state = StateDetector()
        f = canvas()
        paste(f, state.results, 24, 12)
        f = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
        self.assertEqual(state.detect(f)["name"], "resultados")

    def test_results_clears_ids_and_returns_to_menu(self):
        engine = SessionAnalyzer()
        f = canvas()
        paste(f, engine.states.hud, 40, 20)
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
        cv2.rectangle(f, (600, 286), (672, 294), (30, 220, 70), -1)
        cv2.putText(
            f, "7560", (610, 281), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        _, d = engine.analyze(f)
        first = d["tracked_objects"][0]["track_id"]
        result = canvas()
        paste(result, engine.states.results, 24, 12)
        for _ in range(3):
            _, d = engine.analyze(result)
        self.assertEqual(d["game_state"]["name"], "resultados")
        menu = canvas()
        paste(menu, engine.states.menu, 1040, 600)
        for _ in range(12):
            _, d = engine.analyze(menu)
        self.assertEqual(d["game_state"]["name"], "pre_partida")
        for _ in range(6):
            _, d = engine.analyze(f)
        self.assertNotEqual(first, d["tracked_objects"][0]["track_id"])

    def test_invalid_channels_and_dtype_rejected(self):
        for bad in (
            np.zeros((240, 320, 4), np.uint8),
            np.zeros((240, 320, 3), np.float32),
        ):
            with self.assertRaises(ValueError):
                analyze(bad)


class ReplayAndReportTests(unittest.TestCase):
    def test_replay_creates_report_events_and_results_state(self):
        from replay_video import replay

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "sample.avi"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 720)
            )
            self.assertTrue(writer.isOpened())
            detector = StateDetector()
            f = canvas()
            paste(f, detector.results, 24, 12)
            for _ in range(6):
                writer.write(f)
            writer.release()
            output = Path(temp) / "out"
            result = replay(source, output, annotated_input=True)
            self.assertEqual(result["analyzed_frames"], 6)
            self.assertEqual(result["state_frame_counts"], {"resultados": 6})
            self.assertTrue(result["source_is_annotated"])
            self.assertTrue((output / "relatorio.html").exists())
            self.assertEqual(
                len((output / "replay.jsonl").read_text(encoding="utf-8").splitlines()),
                6,
            )

    def test_report_counts_player_evidence_and_filter_rejections(self):
        from brawlla.metrics import SessionMetrics

        m = SessionMetrics()
        m.observe(
            0,
            {
                "game_state": {"name": "combate"},
                "player_health_bar_candidate": {
                    "player_evidence": "nome_verde_e_barra"
                },
                "red_bar_filter_rejections": {"missing_number_evidence": 3},
            },
        )
        report = m.summary()
        self.assertEqual(
            report["player_observations_by_evidence"], {"nome_verde_e_barra": 1}
        )
        self.assertEqual(
            report["red_bar_filter_rejections"], {"missing_number_evidence": 3}
        )


if __name__ == "__main__":
    unittest.main()
