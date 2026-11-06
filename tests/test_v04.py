import unittest
import cv2
import numpy as np
from brawlla.state import StateDetector
from brawlla.objects import ObjectClassifier
from brawlla.tracking import TargetTracker
from brawlla.engine import SessionAnalyzer


def bg():
    return np.full((634, 1280, 3), (100, 55, 105), dtype=np.uint8)


def paste(target, sample, x, y):
    h, w = sample.shape[:2]
    target[y : y + h, x : x + w] = sample


class StateTests(unittest.TestCase):
    def test_absent_hud_is_unknown_without_positive_menu(self):
        d = StateDetector().detect(bg())
        self.assertEqual(d["name"], "desconhecido")

    def test_combat_text(self):
        a = bg()
        s = StateDetector()
        paste(a, s.hud, 210, 40)
        self.assertEqual(s.detect(a)["name"], "combate")

    def test_defeat_takes_priority_over_hud(self):
        a = bg()
        s = StateDetector()
        paste(a, s.hud, 210, 40)
        paste(a, s.defeat, 600, 140)
        self.assertEqual(s.detect(a)["name"], "derrota")

    def test_menu_transition_does_not_hold_old_target_ids(self):
        engine = SessionAnalyzer()
        a = bg()
        paste(a, engine.states.hud, 210, 40)
        name_width = cv2.getTextSize("MEU PLAYER", cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[
            0
        ][0]
        cv2.putText(
            a,
            "MEU PLAYER",
            (492 - name_width // 2, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (30, 230, 50),
            2,
        )
        cv2.rectangle(a, (450, 240), (535, 251), (30, 210, 90), -1)
        cv2.putText(
            a, "7560", (460, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        _, d = engine.analyze(a)
        self.assertEqual(d["game_state"]["name"], "combate")
        self.assertGreater(len(d["tracked_objects"]), 0)
        b = bg()
        paste(b, engine.states.menu, 1040, 530)
        # As mudanças de estado requerem votos consecutivos.
        for _ in range(12):
            _, d = engine.analyze(b)
        self.assertEqual(d["game_state"]["name"], "pre_partida")
        self.assertEqual(d["tracked_objects"], [])


class ObjectTests(unittest.TestCase):
    def test_box_candidate_from_visual_example(self):
        a = bg()
        c = ObjectClassifier()
        cv2.rectangle(a, (350, 350), (416, 358), (48, 40, 235), -1)
        # Abaixo da barra, uma caixa de exemplo fornecida na amostra.
        b = cv2.imread(
            str(
                __import__("pathlib").Path(__file__).resolve().parents[1]
                / "brawlla/assets/caixa_exemplo.png"
            )
        )
        paste(a, b, 350, 363)
        r = c.classify(a, {"x": 350, "y": 350, "width": 66, "height": 9})
        self.assertEqual(r["kind"], "caixa_provavel")
        self.assertGreaterEqual(r["box_evidence"], 0.35)

    def test_black_background_not_box(self):
        a = np.zeros((634, 1280, 3), dtype=np.uint8)
        r = ObjectClassifier().classify(
            a, {"x": 350, "y": 350, "width": 66, "height": 9}
        )
        self.assertNotEqual(r["kind"], "caixa_provavel")


class TrackingTests(unittest.TestCase):
    def test_stable_ids_for_small_motion(self):
        t = TargetTracker()
        a = [{"kind": "personagem_possivel", "center": [200, 150]}]
        b = [{"kind": "personagem_possivel", "center": [213, 151]}]
        self.assertEqual(t.update(a)[0]["track_id"], t.update(b)[0]["track_id"])

    def test_new_ids_not_reused_after_reset(self):
        t = TargetTracker()
        x = [{"kind": "caixa_provavel", "center": [10, 10]}]
        first = t.update(x)[0]["track_id"]
        t.reset()
        self.assertNotEqual(first, t.update(x)[0]["track_id"])

    def test_objects_are_not_linked_twice_same_frame(self):
        t = TargetTracker()
        objects = [
            {"kind": "personagem_possivel", "center": [100, 100]},
            {"kind": "personagem_possivel", "center": [110, 100]},
        ]
        ids = [o["track_id"] for o in t.update(objects)]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
