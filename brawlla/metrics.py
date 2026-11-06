"""Métricas de observação, sem confundir frequência de detecção com acurácia."""

from __future__ import annotations
from collections import Counter, deque
import json
import math
from pathlib import Path


def percentile95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 3)


class SessionMetrics:
    def __init__(self):
        self.frames = 0
        self.combat_frames = 0
        self.player_frames = 0
        self.unknown_frames = 0
        self.states = Counter()
        self.seconds = Counter()
        self.kinds = Counter()
        self.player_methods = Counter()
        self.bar_filter_counts = Counter()
        self.tracks = Counter()
        self.player_tracks = set()
        self.track_event_counts = Counter()
        self.transition_reasons = Counter()
        self.track_timeline = []
        self.track_events_omitted = 0
        self.match_open = False
        self.match_frames = 0
        self.match_player_frames = 0
        self.match_duration = 0.0
        self.match_suspended_frames = 0
        self.match_suspended_s = 0.0
        self.uncertain_frames = 0
        self.previous_match_open = False
        self.previous_suspended = False
        self.timings = {}
        self.first = None
        self.last = None
        self.previous_state = None
        self.timeline = []
        self.timeline_omitted = 0

    def observe(self, elapsed_s: float, data: dict) -> None:
        elapsed_s = float(elapsed_s)
        if (
            not math.isfinite(elapsed_s)
            or elapsed_s < 0
            or (self.last is not None and elapsed_s < self.last)
        ):
            raise ValueError("Tempo de sessão inválido ou fora de ordem")
        name = data.get("game_state", {}).get("name", "desconhecido")
        if self.first is None:
            self.first = elapsed_s
        if self.last is not None:
            interval = elapsed_s - self.last
            self.seconds[self.previous_state] += interval
            if self.previous_match_open:
                self.match_duration += interval
            if self.previous_suspended:
                self.match_suspended_s += interval
        if name != self.previous_state:
            self.transition_reasons[
                data.get("game_state", {}).get("reason", "legado")
            ] += 1
            if len(self.timeline) < 1000:
                self.timeline.append({"elapsed_s": round(elapsed_s, 3), "state": name})
            else:
                self.timeline_omitted += 1
        if name == "combate":
            self.match_open = True
        elif name in ("pre_partida", "derrota", "resultados"):
            self.match_open = False
        suspended = self.match_open and name != "combate"
        if self.match_open:
            self.match_frames += 1
            self.match_player_frames += int(
                bool(data.get("player_health_bar_candidate"))
            )
        self.match_suspended_frames += int(suspended)
        self.uncertain_frames += int(
            name == "combate" and data.get("game_state", {}).get("uncertain", False)
        )
        for event in data.get("track_events", []):
            self.track_event_counts[
                event.get("event", "unknown") + ":" + event.get("reason", "unknown")
            ] += 1
            if len(self.track_timeline) < 500:
                self.track_timeline.append(dict(event))
            else:
                self.track_events_omitted += 1
        self.frames += 1
        self.states[name] += 1
        objects = data.get("tracked_objects", [])
        if name == "combate":
            self.combat_frames += 1
            player = data.get("player_health_bar_candidate")
            self.player_frames += int(bool(player))
            if player:
                self.player_methods[player.get("player_evidence", "legado")] += 1
        for reason, count in data.get(
            "health_bar_filter_rejections", data.get("red_bar_filter_rejections", {})
        ).items():
            if isinstance(count, int) and count >= 0:
                self.bar_filter_counts[reason] += count
        self.unknown_frames += int(any(o.get("kind") == "indefinido" for o in objects))
        for obj in objects:
            self.kinds[obj.get("kind", "indefinido")] += 1
            if "track_id" in obj:
                self.tracks[obj["track_id"]] += 1
                if obj.get("kind") == "jogador_candidato":
                    self.player_tracks.add(obj["track_id"])
        timings = {
            **data.get("timings_ms", {}),
            **{
                "pipeline." + key: value
                for key, value in data.get("pipeline_timings_ms", {}).items()
            },
        }
        for key, value in timings.items():
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                item = self.timings.setdefault(
                    key,
                    {"count": 0, "sum": 0.0, "max": 0.0, "recent": deque(maxlen=2048)},
                )
                item["count"] += 1
                item["sum"] += value
                item["max"] = max(item["max"], value)
                item["recent"].append(value)
        self.previous_match_open = self.match_open
        self.previous_suspended = suspended
        self.last, self.previous_state = elapsed_s, name

    def summary(self) -> dict:
        duration = 0.0 if self.first is None else self.last - self.first
        return {
            "analyzed_frames": self.frames,
            "unique_player_track_ids": len(self.player_tracks),
            "inferred_match_frames": self.match_frames,
            "inferred_match_duration_s": round(self.match_duration, 3),
            "inferred_match_frames_with_player_candidate": self.match_player_frames,
            "inferred_match_candidate_coverage_pct": (
                round(100 * self.match_player_frames / self.match_frames, 2)
                if self.match_frames
                else None
            ),
            "analysis_suspended_in_match_frames": self.match_suspended_frames,
            "analysis_suspended_in_match_s": round(self.match_suspended_s, 3),
            "uncertain_combat_frames": self.uncertain_frames,
            "state_transition_reasons": dict(self.transition_reasons),
            "track_event_counts": dict(self.track_event_counts),
            "track_event_timeline": self.track_timeline.copy(),
            "track_events_omitted": self.track_events_omitted,
            "observation_duration_s": round(duration, 3),
            "analysis_fps_over_observation": (
                round((self.frames - 1) / duration, 2) if duration > 0 else 0.0
            ),
            "state_frame_counts": dict(self.states),
            "state_duration_s": {k: round(v, 3) for k, v in self.seconds.items()},
            "state_transitions": max(0, len(self.timeline) + self.timeline_omitted - 1),
            "combat_frames": self.combat_frames,
            "combat_frames_with_player_candidate": self.player_frames,
            "player_candidate_coverage_pct": (
                round(100 * self.player_frames / self.combat_frames, 2)
                if self.combat_frames
                else None
            ),
            "player_observations_by_evidence": dict(self.player_methods),
            "red_bar_filter_rejections": dict(self.bar_filter_counts),
            "health_bar_filter_rejections": dict(self.bar_filter_counts),
            "frames_with_undefined_objects": self.unknown_frames,
            "object_observations_by_kind": dict(self.kinds),
            "unique_track_ids": len(self.tracks),
            "track_ids_seen_once": sum(n == 1 for n in self.tracks.values()),
            "timings_ms": {
                key: {
                    "mean": round(v["sum"] / v["count"], 3),
                    "max": round(v["max"], 3),
                    "samples": v["count"],
                    "p95_recent": percentile95(v["recent"]),
                    "p95_window_samples": len(v["recent"]),
                }
                for key, v in self.timings.items()
            },
            "timeline": self.timeline.copy(),
            "timeline_transitions_omitted": self.timeline_omitted,
            "limitations": [
                "Cobertura indica presença de candidato, não acurácia nem identificação confirmada.",
                "Janela de partida é inferida dos estados; inclui desconhecido entre combate e menu/derrota/resultados. Não é anotação humana.",
                "IDs são trilhas de barras na tela, não número de brawlers ou eliminações.",
                "Duração termina na última análise e atribui cada intervalo ao estado anterior.",
                "p95 usa até as últimas 2048 medições de cada etapa; média e máximo usam a sessão inteira.",
            ],
        }


def summarize_log(path: Path) -> dict:
    metrics = SessionMetrics()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                metrics.observe(row["elapsed_s"], row["detections"])
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"JSONL inválido na linha {line_number}: {exc}"
                ) from exc
    return metrics.summary()


def save_report(summary, destination):
    """Compatibilidade; novos consumidores importam reporting.save_report."""
    from .reporting import save_report as write

    return write(summary, destination)
