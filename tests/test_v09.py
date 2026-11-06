"""Vida crítica, identidade e movimento com câmera; sem depender de BlueStacks."""

from dataclasses import replace
import unittest

import cv2
import numpy as np

from brawlla.config import VisionConfig, AnalyzerConfig
from brawlla.engine import SessionAnalyzer
from brawlla.health import number_evidence
from brawlla.motion import MotionEvidence
from brawlla.player import PlayerLocator
from brawlla.vision import detect


def sample(
    fill=15, color=(0, 0, 245), name=True, numbers=True, plate=(95, 45, 45), offset=0
):
    frame = np.full((720, 1280, 3), (100, 55, 105), np.uint8)
    cx = 640 + offset
    if name:
        label = "MEU PLAYER"
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv2.putText(
            frame,
            label,
            (cx - tw // 2, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 230, 50),
            2,
        )
    cv2.rectangle(frame, (cx - 40, 226), (cx + 40, 239), plate, -1)
    if fill:
        cv2.rectangle(frame, (cx - 37, 229), (cx - 38 + fill, 235), color, -1)
    if numbers:
        cv2.putText(
            frame,
            "1900",
            (cx - 19, 223),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
        )
    return frame


def obj(cx=640, cy=290, kind="indefinido"):
    return {
        "x": cx - 38,
        "y": cy - 5,
        "width": 76,
        "height": 10,
        "center": [cx, cy],
        "kind": kind,
        "confidence": 0.4,
    }


class HealthTests(unittest.TestCase):
    def test_one_pixel_fill_in_all_supported_colors(self):
        for color in [
            (30, 230, 50),
            (20, 220, 245),
            (0, 0, 245),
            (245, 245, 245),
            (230, 180, 30),
            (240, 40, 210),
        ]:
            with self.subTest(color=color):
                result, _ = detect(sample(fill=1, color=color))
                player = result["player_health_bar_candidate"]
                # Flash branco de 1px sozinho não basta; o recipiente observado deve sustentar a decisão.
                self.assertIsNotNone(player)
                self.assertLess(abs(player["center"][0] - 640), 8)
                self.assertTrue(player["health_evidence"]["current_frame_observed"])

    def test_single_hp_digit_with_tiny_fill_and_strong_container(self):
        frame = sample(fill=1, numbers=False)
        cv2.putText(
            frame, "1", (636, 223), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1
        )
        player = detect(frame)[0]["player_health_bar_candidate"]
        self.assertIsNotNone(player)
        self.assertEqual(player["number_evidence"]["glyphs"], 1)
        self.assertTrue(player["health_evidence"]["container"])

    def test_critical_blue_container_with_hp_glyphs(self):
        result, _ = detect(sample(fill=2))
        player = result["player_health_bar_candidate"]
        self.assertIsNotNone(player)
        self.assertGreater(player["container_dark_fraction"], 0.3)
        self.assertGreater(player["number_evidence"]["glyphs"], 0)
        self.assertEqual(player["bar_color"], "vermelha")

    def test_full_container_does_not_shrink_with_fill(self):
        centers = []
        for fill in (70, 35, 15, 2):
            d, _ = detect(sample(fill=fill))
            centers.append(d["player_health_bar_candidate"]["center"])
        self.assertLess(max(p[0] for p in centers) - min(p[0] for p in centers), 4)

    def test_glowing_effect_attached_below_name_does_not_erase_text_band(self):
        frame = sample()
        cv2.rectangle(frame, (585, 193), (589, 235), (30, 230, 50), -1)
        result, _ = detect(frame)
        self.assertIsNotNone(result["player_health_bar_candidate"])

    def test_green_bar_digits_without_name_never_assign_player(self):
        result, _ = detect(sample(color=(30, 230, 50), name=False))
        self.assertIsNone(result["player_health_bar_candidate"])

    def test_small_green_fragment_and_hud_are_not_name(self):
        frame = sample(name=False)
        cv2.putText(
            frame, "ABC", (625, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 230, 50), 1
        )
        result, _ = detect(frame)
        self.assertIsNone(result["player_health_bar_candidate"])
        # Nome/barras reais deslocados para o HUD não identificam o jogador.
        shifted = cv2.warpAffine(
            sample(),
            np.float32([[1, 0, -480], [0, 1, -145]]),
            (1280, 720),
            borderValue=(100, 55, 105),
        )
        result, _ = detect(shifted)
        self.assertIsNone(result["player_health_bar_candidate"])

    def test_solid_white_patch_does_not_count_as_hp_digits(self):
        frame = sample(numbers=False)
        cv2.rectangle(frame, (616, 213), (664, 224), (245, 245, 245), -1)
        result, _ = detect(frame)
        self.assertIsNone(result["player_health_bar_candidate"])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.assertEqual(number_evidence(hsv, 600, 229, 80)["glyphs"], 0)

    def test_position_jump_rejected_without_teleporting_identity(self):
        locator = PlayerLocator()
        first, _ = detect(sample(), locator, 0)
        self.assertIsNotNone(first["player_health_bar_candidate"])
        jump, _ = detect(sample(offset=350), locator, 0.067)
        self.assertIsNone(jump["player_health_bar_candidate"])
        back, _ = detect(sample(fill=2), locator, 0.134)
        self.assertIsNotNone(back["player_health_bar_candidate"])
        self.assertTrue(back["player_health_bar_candidate"]["name_similarity"] > 0.5)

    def test_name_reference_survives_gap_but_no_prediction_is_emitted(self):
        locator = PlayerLocator()
        detect(sample(), locator, 0)
        blank = np.full((720, 1280, 3), (100, 55, 105), np.uint8)
        self.assertIsNone(detect(blank, locator, 1)[0]["player_health_bar_candidate"])
        self.assertIsNone(locator.previous)
        self.assertIsNotNone(locator.reference_name)
        self.assertIsNotNone(
            detect(sample(), locator, 1.1)[0]["player_health_bar_candidate"]
        )

    def test_all_resolutions_support_numbers_and_critical_fill(self):
        for width in (640, 960, 1280, 1920):
            with self.subTest(width=width):
                frame = cv2.resize(sample(fill=3), (width, round(width * 720 / 1280)))
                result, _ = detect(frame)
                self.assertIsNotNone(result["player_health_bar_candidate"])

    def test_ammo_without_health_glyphs_is_not_hp(self):
        frame = sample(fill=0, numbers=False)
        cv2.rectangle(frame, (604, 242), (624, 250), (0, 160, 250), -1)
        result, _ = detect(frame)
        self.assertIsNone(result["player_health_bar_candidate"])

    def test_partial_other_bar_requires_digits_and_background(self):
        frame = sample(name=False, fill=3)
        d, _ = detect(frame)
        self.assertGreaterEqual(len(d["other_health_bar_candidates"]), 1)
        self.assertIsNone(d["player_health_bar_candidate"])
        blank = sample(name=False, fill=3, numbers=False)
        d, _ = detect(blank)
        self.assertEqual(d["other_health_bar_candidates"], [])

    def test_other_bar_colors_do_not_change_red_alias_semantics(self):
        d, _ = detect(sample(name=False, fill=60, color=(20, 220, 245)))
        self.assertTrue(d["other_health_bar_candidates"])
        self.assertEqual(d["red_health_bar_candidates"], [])

    def test_low_fill_is_proxy_and_dark_container_is_not_exact_hp(self):
        player = detect(sample(fill=2))[0]["player_health_bar_candidate"]
        self.assertEqual(player["health_status"], "low_fill_candidate")
        self.assertLess(player["fill_fraction_proxy"], 0.1)
        self.assertNotIn("hp", player)

    def test_motion_can_be_disabled_without_losing_visual_detection(self):
        config = AnalyzerConfig(vision=replace(VisionConfig(), motion_enabled=False))
        engine = SessionAnalyzer(config)
        frame = sample(fill=2)
        hud = engine.states.hud
        frame[20 : 20 + hud.shape[0], 40 : 40 + hud.shape[1]] = hud
        _, result = engine.analyze(frame, 0)
        self.assertIsNotNone(result["player_health_bar_candidate"])
        self.assertFalse(result["camera_motion"]["valid"])
        self.assertEqual(
            result["camera_motion"]["reason"], "disabled_or_outside_combat"
        )

    def test_config_validates_name_thresholds(self):
        for options in (
            {"player_name_min_width_px": -1},
            {"player_jump_base_px": 0},
            {"player_name_similarity_min": 1.1},
            {"motion_enabled": 1},
        ):
            with self.assertRaises(ValueError):
                replace(VisionConfig(), **options)


class MotionTests(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(123)
        self.background = np.random.default_rng(123).integers(
            25, 180, (720, 1280, 3), dtype=np.uint8
        )
        self.body = np.random.default_rng(999).integers(
            0, 255, (70, 90, 3), dtype=np.uint8
        )

    def scene(self, pan=0, move=0):
        frame = cv2.warpAffine(
            self.background,
            np.float32([[1, 0, pan], [0, 1, 0]]),
            (1280, 720),
            borderMode=cv2.BORDER_REFLECT,
        )
        cx = 640 + pan + move
        frame[310:380, cx - 45 : cx + 45] = self.body
        return frame, cx

    def test_pan_does_not_turn_stationary_bar_into_moving_character(self):
        motion = MotionEvidence()
        last = None
        for i in range(8):
            frame, cx = self.scene(pan=i * 4)
            objects = [obj(cx)]
            camera = motion.update(frame, objects, i * 0.1)
            last = objects[0]
        self.assertTrue(camera["valid"])
        self.assertEqual(
            last["motion_evidence"]["classification"], "stationary_relative_camera"
        )
        self.assertEqual(last["kind"], "indefinido")
        self.assertLess(last["motion_evidence"]["relative_camera_speed_px_s"], 12)

    def test_moving_bar_and_body_support_character(self):
        motion = MotionEvidence()
        for i in range(8):
            frame, cx = self.scene(move=i * 8)
            objects = [obj(cx)]
            camera = motion.update(frame, objects, i * 0.1)
        self.assertTrue(camera["valid"])
        self.assertTrue(objects[0]["motion_evidence"]["body_bar_consistent"])
        self.assertEqual(
            objects[0]["motion_evidence"]["classification"], "moving_with_body"
        )
        self.assertEqual(objects[0]["kind"], "personagem_possivel")

    def test_stationary_character_does_not_become_box(self):
        motion = MotionEvidence()
        for i in range(8):
            frame, cx = self.scene(pan=i * 3)
            objects = [obj(cx, kind="personagem_possivel")]
            motion.update(frame, objects, i * 0.1)
        self.assertEqual(objects[0]["kind"], "personagem_possivel")

    def test_textureless_camera_stays_uncertain(self):
        motion = MotionEvidence()
        flat = np.full((720, 1280, 3), 80, np.uint8)
        for i in range(4):
            objects = [obj()]
            camera = motion.update(flat, objects, i * 0.1)
        self.assertFalse(camera["valid"])
        self.assertFalse(objects[0]["motion_evidence"]["camera_valid"])

    def test_bar_alone_moving_without_body_does_not_promote_character(self):
        motion = MotionEvidence()
        for i in range(8):
            frame, _ = self.scene()
            objects = [obj(640 + i * 7)]
            motion.update(frame, objects, i * 0.1)
        self.assertNotEqual(
            objects[0]["motion_evidence"]["classification"], "moving_with_body"
        )
        self.assertEqual(objects[0]["kind"], "indefinido")

    def test_reset_clears_temporal_evidence(self):
        motion = MotionEvidence()
        frame, cx = self.scene()
        motion.update(frame, [obj(cx)], 0)
        motion.reset()
        objects = [obj(cx)]
        camera = motion.update(frame, objects, 1)
        self.assertFalse(camera["valid"])
        self.assertEqual(
            objects[0]["motion_evidence"]["classification"], "insufficient_evidence"
        )


if __name__ == "__main__":
    unittest.main()
