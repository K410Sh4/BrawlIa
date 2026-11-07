"""Testes de contrato: apresentação, associação e isolamento de partida."""

from types import SimpleNamespace
import json
import unittest

import cv2
import numpy as np

from brawlla.config import AnalyzerConfig, RosterConfig
from brawlla.roster import MatchRoster, normalize_name, name_regions, roster_visible


def lineup():
    frame = np.full((720, 1280, 3), 125, dtype=np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(name_regions(frame)):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (22, 22, 22), -1)
        cv2.putText(
            frame, f"NICK{i+1}", (x1 + 7, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1,
        )
    return frame


class RosterTests(unittest.TestCase):
    def test_layout_detected_before_combat(self):
        frame = lineup()
        self.assertTrue(roster_visible(frame))
        self.assertFalse(roster_visible(np.full_like(frame, 170)))
        self.assertEqual(len(name_regions(frame)), 10)

    def test_text_normalization_and_config_validation(self):
        self.assertEqual(normalize_name(" YT | Sky_Zynox "), "ytskyzynox")
        with self.assertRaises(ValueError):
            RosterConfig(roster_votes=0)
        with self.assertRaises(ValueError):
            RosterConfig(match_threshold=1.5)
        self.assertIsInstance(AnalyzerConfig().roster, RosterConfig)

    def test_queued_roster_only_creates_pending_slots(self):
        frame = lineup()
        roster = MatchRoster(RosterConfig())
        roster._send = lambda *_: True
        result = roster.observe(frame, "desconhecido", 0.0, [])
        self.assertEqual(result["status"], "detected")
        self.assertEqual(result["confirmed_names"], 0)
        self.assertEqual(len(result["participants"]), 10)
        self.assertIsNone(result["local_participant_id"])
        self.assertEqual(result["participants"][0]["participant_id"], "P01")

    def test_names_require_two_agreeing_ocr_observations(self):
        roster = MatchRoster(RosterConfig())
        roster.detected = True
        reads = [(f"Nick{i+1}", 0.92) for i in range(10)]
        roster.completed.put((roster.generation, "roster", reads))
        roster._poll()
        self.assertEqual(roster.snapshot()["confirmed_names"], 0)
        roster.completed.put((roster.generation, "roster", reads))
        roster._poll()
        self.assertEqual(roster.snapshot()["confirmed_names"], 10)
        self.assertEqual(roster.participants[0]["name"], "Nick1")

    def test_unique_match_requires_repeated_evidence(self):
        roster = MatchRoster(RosterConfig())
        roster.detected = True
        roster.participants[0].update(name="SkyZynoxCrow", status="confirmed")
        roster.participants[1].update(name="OtherPlayer", status="confirmed")
        roster._match_track(42, "SkyZynoxCrow", 0.98)
        self.assertEqual(roster.bindings, {})
        roster._match_track(42, "SkyZynoxCrow", 0.98)
        self.assertEqual(roster.bindings, {42: "P01"})
        objs = [{"track_id":42,"kind":"jogador_candidato"}]
        result = roster.observe(np.zeros((720,1280,3),dtype=np.uint8), "combate", 1, objs)
        self.assertEqual(result["local_participant_id"], "P01")
        self.assertEqual(objs[0]["participant_id"], "P01")
        json.dumps(result)

    def test_ambiguous_names_cannot_claim_participant(self):
        roster = MatchRoster(RosterConfig())
        roster.participants[0].update(name="John", status="confirmed")
        roster.participants[1].update(name="John", status="confirmed")
        roster._match_track(42, "John", 0.99)
        roster._match_track(42, "John", 0.99)
        self.assertFalse(roster.bindings)

    def test_dormant_old_track_can_be_rebound(self):
        roster = MatchRoster(RosterConfig())
        roster.participants[0].update(name="SkyZynoxCrow", status="confirmed")
        roster.bindings = {1:"P01"}
        roster.active_track_ids = {1}
        for _ in range(2):
            roster._match_track(99, "SkyZynoxCrow", 0.96)
        self.assertEqual(roster.bindings, {1:"P01"})
        roster.active_track_ids = set()
        roster._match_track(99, "SkyZynoxCrow", 0.96)
        self.assertEqual(roster.bindings, {99:"P01"})

    def test_second_match_does_not_inherit_roster(self):
        roster = MatchRoster(RosterConfig())
        roster.participants[0].update(name="Nick1", status="confirmed")
        roster.bindings[42] = "P01"
        roster.had_combat = True
        roster._send = lambda *_: True
        result = roster.observe(lineup(), "desconhecido", 12, [])
        self.assertEqual(result["status"], "detected")
        self.assertEqual(result["confirmed_names"], 0)
        self.assertIsNone(result["local_participant_id"])
        self.assertFalse(roster.bindings)

    def test_ocr_response_contract(self):
        from brawlla.roster import read_name
        fake = lambda _: SimpleNamespace(txts=("Nick1",), scores=(0.99,))
        answer = read_name(fake, np.zeros((20, 80, 3), np.uint8))
        self.assertEqual(answer, ("Nick1", 0.99))


if __name__ == "__main__":
    unittest.main()
