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
    name_key: str | None = None
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
        self._dormant_names = {}
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
        self._dormant_names.clear()
        self._width = None
        # IDs e relógio não são reiniciados dentro da mesma sessão.

    @staticmethod
    def _compatible(a, b):
        return (a == "jogador_candidato") == (b == "jogador_candidato")

    def _supported_name(self, candidate):
        """Aceita âncora visual apenas do jogador e com evidência forte.

        Um nome semelhante em dois candidatos simultâneos é ambíguo; a
        decisão de um para um é feita no update, sem forçar a mesma trilha.
        """
        if candidate.kind != "jogador_candidato":
            return None
        key = candidate.properties.get("name_identity_key")
        score = candidate.properties.get("name_similarity")
        if (
            not isinstance(key, str)
            or not key.startswith("visual:")
            or len(key) != len("visual:") + 20
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            or score < self.config.name_reid_min_similarity
            or score > 1
        ):
            return None
        return key

    def update(
        self, objects, frame_width=1280, timestamp_s=None, *, camera_motion=None
    ):
        if frame_width <= 0:
            raise ValueError("Largura inválida")
        now = self.clock.resolve(timestamp_s)
        if self._width is not None and self._width != frame_width:
            self.reset("resolucao_alterada", now)
        self._width = frame_width
        for key, track in list(self._dormant_names.items()):
            if now - track.observed_at > self.config.name_reid_max_age_s:
                del self._dormant_names[key]
        for tid, track in list(self._tracks.items()):
            if now - track.observed_at > self.config.max_absence_s + 1e-9:
                self._event("lost", now, "tempo_sem_observacao_excedido", track)
                if track.name_key:
                    self._dormant_names[track.name_key] = track
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
        # Reserva identidades confirmadas visualmente antes da associação
        # geométrica. Nunca repete uma observação quando o nome desaparece.
        keys = [self._supported_name(obj) for obj in candidates]
        counts = Counter(k for k in keys if k is not None)
        assignments = {}
        name_assigned = set()
        used = set()
        for index, key in enumerate(keys):
            if key is None or counts[key] != 1:
                continue
            cx, cy = candidates[index].center
            nearby = [
                track
                for track in self._tracks.values()
                if track.name_key == key
                and self._compatible(track.kind, candidates[index].kind)
                and math.hypot(track.x - cx, track.y - cy)
                <= self.config.name_reid_max_distance_px * scale
            ]
            if nearby:
                match = min(
                    nearby, key=lambda t: (math.hypot(t.x - cx, t.y - cy), -t.hits)
                )
                assignments[index] = match.id
                name_assigned.add(index)
                used.add(match.id)
                continue
            dormant = self._dormant_names.get(key)
            if (
                dormant is not None
                and now - dormant.observed_at <= self.config.name_reid_max_age_s
                and math.hypot(dormant.x - cx, dormant.y - cy)
                <= self.config.name_reid_max_distance_px * scale
                and dormant.id not in used
            ):
                self._tracks[dormant.id] = dormant
                del self._dormant_names[key]
                assignments[index] = dormant.id
                name_assigned.add(index)
                used.add(dormant.id)
                self._event("reidentified", now, "nome_visual_confirmado", dormant)

        for index, obj in enumerate(candidates):
            cx, cy = obj.center
            for track in self._tracks.values():
                if track.id in used or not self._compatible(track.kind, obj.kind):
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
                # Um salto corrigido pelo nome não é velocidade do brawler.
                # Redefinir a velocidade evita propagar o outlier anterior.
                if index in name_assigned:
                    px = track.x + track.vx * min(
                        max(0, dt), self.config.prediction_horizon_s
                    )
                    py = track.y + track.vy * min(
                        max(0, dt), self.config.prediction_horizon_s
                    )
                    if math.hypot(cx - px, cy - py) > self.config.max_gate_px * scale:
                        track.vx = track.vy = 0.0
                        track.screen_vx = track.screen_vy = 0.0
                        track.hits = max(track.hits, 2)
                        dt = 0.0
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
            if keys[index] and counts[keys[index]] == 1:
                # A identidade nunca depende do preenchimento/cor da barra.
                if track.name_key in (None, keys[index]):
                    track.name_key = keys[index]
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
                    "identity_association": (
                        "nome_visual" if index in name_assigned else "geometria"
                    ),
                    "track_name_identity_key": track.name_key,
                    "name_is_ocr": False,
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
