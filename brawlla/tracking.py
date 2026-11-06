"""Rastreamento por timestamps, com eventos explícitos de criação/perda/reset."""

from collections import Counter, deque
from dataclasses import dataclass, field, replace
import math
import numpy as np
from .config import TrackingConfig
from .domain import Candidate, Timeline, TrackEvent


@dataclass
class _Track:
    id: int
    kind: str
    x: float
    y: float
    observed_at: float
    hits: int = 1
    vx: float = 0.0
    vy: float = 0.0
    observed_x: float = 0.0
    observed_y: float = 0.0
    screen_vx: float = 0.0
    screen_vy: float = 0.0
    compensated: bool = False
    kinds: deque = field(default_factory=lambda: deque(maxlen=5))


class TargetTracker:
    def __init__(self, max_missed=None, distance_px=None, *, config=None):
        self.config = config or TrackingConfig()
        # Compatibilidade: parâmetros antigos são convertidos uma única vez a unidades temporais.
        if max_missed is not None:
            if max_missed < 0:
                raise ValueError("Limite de ausência inválido")
            self.config = replace(self.config, max_absence_s=max(1e-6, max_missed / 15))
        if distance_px is not None:
            if distance_px <= 0:
                raise ValueError("Distância inválida")
            self.config = replace(
                self.config,
                base_distance_px=distance_px,
                max_gate_px=max(distance_px, self.config.max_gate_px),
            )
        self._tracks = {}
        self._next_id = 1
        self._width = None
        self.clock = Timeline()
        self._events = deque(maxlen=500)
        self._events_omitted = 0

    def _event(self, event, now, reason, track=None):
        if len(self._events) == self._events.maxlen:
            self._events_omitted += 1
        self._events.append(
            TrackEvent(
                event,
                now,
                reason,
                track.id if track else None,
                track.kind if track else None,
            ).to_dict()
        )

    def drain_events(self):
        events = list(self._events)
        self._events.clear()
        if self._events_omitted:
            events.append(
                {
                    "event": "omitted",
                    "reason": "limite_eventos_pendentes",
                    "timestamp_s": self.clock.last or 0.0,
                    "count": self._events_omitted,
                }
            )
            self._events_omitted = 0
        return events

    def reset(self, reason="reinicio_solicitado", timestamp_s=None):
        now = (
            self.clock.resolve(timestamp_s)
            if timestamp_s is not None
            else (self.clock.last or 0.0)
        )
        for track in self._tracks.values():
            self._event("lost", now, reason, track)
        self._event("reset", now, reason)
        self._tracks.clear()
        self._width = None
        # IDs e relógio não são reiniciados dentro da mesma sessão.

    @staticmethod
    def _compatible(a, b):
        return (a == "jogador_candidato") == (b == "jogador_candidato")

    def update(
        self, objects, frame_width=1280, timestamp_s=None, *, camera_motion=None
    ):
        if frame_width <= 0:
            raise ValueError("Largura inválida")
        now = self.clock.resolve(timestamp_s)
        if self._width is not None and self._width != frame_width:
            self.reset("resolucao_alterada", now)
        self._width = frame_width
        for tid, track in list(self._tracks.items()):
            if now - track.observed_at > self.config.max_absence_s + 1e-9:
                self._event("lost", now, "tempo_sem_observacao_excedido", track)
                del self._tracks[tid]
        affine = None
        if camera_motion and camera_motion.get("valid"):
            raw = camera_motion.get("affine_original")
            if raw is not None:
                try:
                    candidate = np.asarray(raw, dtype=float)
                    if (
                        candidate.shape == (2, 3)
                        and np.isfinite(candidate).all()
                        and camera_motion.get("resolution", [frame_width])[0]
                        == frame_width
                    ):
                        affine = candidate
                except (TypeError, ValueError):
                    pass
        for track in self._tracks.values():
            if track.compensated != (affine is not None):
                track.vx = track.vy = 0.0
            if affine is not None:
                track.x, track.y = (affine @ np.array([track.x, track.y, 1.0])).tolist()
                track.vx, track.vy = (
                    affine[:, :2] @ np.array([track.vx, track.vy])
                ).tolist()
            track.compensated = affine is not None
        candidates = [Candidate.from_dict(o) for o in objects]
        pairs = []
        scale = frame_width / self.config.reference_width
        for index, obj in enumerate(candidates):
            cx, cy = obj.center
            for track in self._tracks.values():
                if not self._compatible(track.kind, obj.kind):
                    continue
                dt = max(0.0, now - track.observed_at)
                horizon = min(dt, self.config.prediction_horizon_s)
                px, py = track.x + track.vx * horizon, track.y + track.vy * horizon
                distance = math.hypot(px - cx, py - cy)
                gate = (
                    min(
                        self.config.max_gate_px,
                        self.config.base_distance_px
                        + self.config.motion_budget_px_s * dt,
                    )
                    * scale
                )
                if distance < gate:
                    penalty = (
                        0
                        if track.kind == obj.kind
                        or "indefinido" in (track.kind, obj.kind)
                        else gate * 0.2
                    )
                    pairs.append((distance + penalty, track.id, cx, cy, index))
        assignments = {}
        used = set()
        for _, tid, _, _, index in sorted(pairs):
            if index not in assignments and tid not in used:
                assignments[index] = tid
                used.add(tid)
        results = []
        for index, obj in enumerate(candidates):
            cx, cy = obj.center
            if index in assignments:
                track = self._tracks[assignments[index]]
                dt = now - track.observed_at
                if dt > 1e-9:
                    alpha = (
                        1.0
                        if track.hits == 1
                        else 1 - math.exp(-dt / self.config.velocity_tau_s)
                    )
                    track.vx = (1 - alpha) * track.vx + alpha * (cx - track.x) / dt
                    track.vy = (1 - alpha) * track.vy + alpha * (cy - track.y) / dt
                    track.screen_vx = (1 - alpha) * track.screen_vx + alpha * (
                        cx - track.observed_x
                    ) / dt
                    track.screen_vy = (1 - alpha) * track.screen_vy + alpha * (
                        cy - track.observed_y
                    ) / dt
                track.x, track.y, track.observed_at = cx, cy, now
                track.observed_x, track.observed_y = cx, cy
                track.hits += 1
            else:
                compatible = any(
                    self._compatible(t.kind, obj.kind) for t in self._tracks.values()
                )
                reason = (
                    "fora_da_distancia_de_associacao"
                    if compatible
                    else "sem_trilha_compativel"
                )
                track = _Track(
                    self._next_id,
                    obj.kind,
                    cx,
                    cy,
                    now,
                    observed_x=cx,
                    observed_y=cy,
                    compensated=affine is not None,
                )
                self._next_id += 1
                self._tracks[track.id] = track
                self._event("created", now, reason, track)
            track.kinds.append(obj.kind)
            winner, votes = Counter(track.kinds).most_common(1)[0]
            if votes >= 3:
                track.kind = winner
            results.append(
                {
                    **obj.properties,
                    "track_id": track.id,
                    "track_hits": track.hits,
                    "track_kind": track.kind,
                    "screen_motion_px_per_s": [
                        round(track.screen_vx, 2),
                        round(track.screen_vy, 2),
                    ],
                    "association_motion_px_per_s": [
                        round(track.vx, 2),
                        round(track.vy, 2),
                    ],
                    "association_reference": (
                        "camera_compensated" if affine is not None else "screen"
                    ),
                    "tracked_center": [round(track.x), round(track.y)],
                }
            )
        return results
