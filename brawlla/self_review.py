"""Autoavaliação leve de percepção sem gravação de PNG nem autotreino.

Seleciona momentos para revisão manual no vídeo, usando apenas metadados
que o BrawlIA já produziu. Não presume ground truth e não altera pesos.
"""

from collections import Counter
from math import isfinite


class SelfReview:
    def __init__(self, *, max_events=120, min_gap_s=1.5):
        if max_events < 1 or min_gap_s < 0:
            raise ValueError("Parâmetros inválidos para autoavaliação")
        self.max_events = max_events
        self.min_gap_s = min_gap_s
        self.frames_seen = 0
        self.combat_frames = 0
        self.missing_player = 0
        self.events = []
        self.counts = Counter()
        self._last_event = {}

    def _flag(self, reason, elapsed, *, track_id=None):
        self.counts[reason] += 1
        if not isinstance(elapsed, (int, float)) or not isfinite(elapsed):
            return
        if elapsed - self._last_event.get(reason, float("-inf")) < self.min_gap_s:
            return
        self._last_event[reason] = elapsed
        if len(self.events) < self.max_events:
            self.events.append(
                {
                    "at_s": round(max(0, elapsed), 3),
                    "reason": reason,
                    "track_id": track_id,
                    "requires_review": True,
                }
            )

    def observe(self, data):
        """Custo proporcional ao número de alvos, sem operações de imagem."""
        self.frames_seen += 1
        state = data.get("game_state", {})
        combat = state.get("name") == "combate"
        elapsed = data.get("session_elapsed_s", 0)
        if combat:
            self.combat_frames += 1
            player = data.get("player_health_bar_candidate")
            if player is None:
                self.missing_player += 1
                self._flag("player_not_detected", elapsed)
            else:
                similarity = player.get("name_similarity")
                if isinstance(similarity, (int, float)) and similarity < 0.55:
                    self._flag("player_name_low_similarity", elapsed)
                    status = str(player.get("health_status", ""))
                    if "low" in status or "critical" in status:
                        self._flag("low_hp_unconfirmed_identity", elapsed)
        for event in data.get("track_events", []):
            if event.get("event") in ("lost", "reidentified"):
                self._flag(
                    "track_" + event["event"],
                    elapsed,
                    track_id=event.get("track_id"),
                )
        roster = data.get("match_roster", {})
        if combat and roster.get("status") == "detected":
            if roster.get("confirmed_names", 0) == 0:
                self._flag("roster_without_confirmed_names", elapsed)

        timings = data.get("timings_ms", {})
        total = timings.get("total")
        if isinstance(total, (int, float)) and total > 33.33:
            self._flag("processing_over_33ms", elapsed)

    def snapshot(self):
        return {
            "method": "uncertainty_metadata_only",
            "frames_seen": self.frames_seen,
            "combat_frames": self.combat_frames,
            "combat_frames_without_player": self.missing_player,
            "flag_counts": dict(self.counts),
            "review_points": list(self.events),
            "saved_png": False,
            "weights_updated": False,
            "automatic_retraining": False,
            "note": (
                "Eventos são hipóteses para revisão do MP4; sem rótulos "
                "verificados não existe autoaprendizado confiável."
            ),
        }
