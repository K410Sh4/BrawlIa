"""Identidade temporal: observado != não observado != ocluído != perdido."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from .geometry import Rect
from .silhouette import Signature, compare_silhouettes, structural_change_ratio
import numpy as np


@dataclass
class Detection:
    bbox: Rect
    kind: str
    color: str
    confidence: float
    signature: Signature | None = None
    fill_fraction: float | None = None
    source: str = "observed"
    stable_patch: np.ndarray | None = None


@dataclass
class Track:
    track_id: int
    bbox: Rect
    kind: str
    color: str
    confidence: float
    last_seen: float
    velocity: tuple[float, float] = (0.0, 0.0)
    status: str = "observado"
    misses: int = 0
    signature: Signature | None = field(default=None, repr=False)
    fill_fraction: float | None = None
    stable_patch: np.ndarray | None = field(default=None, repr=False)
    structural_change: float | None = None
    structural_streak: int = 0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "xywh": self.bbox.xywh(),
            "kind": self.kind,
            "color": self.color,
            "confidence": round(self.confidence, 3),
            "last_seen": round(self.last_seen, 4),
            "velocity": [round(v, 2) for v in self.velocity],
            "status": self.status,
            "misses": self.misses,
            "fill_fraction": self.fill_fraction,
            "health_exact": None,
            "structural_change": self.structural_change,
            "structural_streak": self.structural_streak,
        }


class ContextMemory:
    def __init__(self, max_absence_s: float = 2.5, max_speed_px_s: float = 650):
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.max_absence_s = max_absence_s
        self.max_speed_px_s = max_speed_px_s
        self.events = []

    def reset(self, reason: str):
        self.events.append({"event": "reset", "reason": reason})
        self.tracks.clear()

    def update(
        self,
        detections: list[Detection],
        observed: list[Rect],
        now: float,
        screen_size: tuple[int, int],
    ) -> list[dict]:
        """Associa detecções com movimento/silhueta, sem inventar evidências."""
        claimed: set[int] = set()
        for det in sorted(detections, key=lambda x: x.confidence, reverse=True):
            best_id, best_score = None, float("-inf")
            for tid, track in self.tracks.items():
                if tid in claimed or track.kind != det.kind:
                    continue
                dt = max(0.001, now - track.last_seen)
                cx, cy = track.bbox.center
                predicted = cx + track.velocity[0] * min(dt, 0.4), cy + track.velocity[
                    1
                ] * min(dt, 0.4)
                dx = det.bbox.center[0] - predicted[0]
                dy = det.bbox.center[1] - predicted[1]
                distance = math.hypot(dx, dy)
                max_distance = max(35, min(200, self.max_speed_px_s * dt + 30))
                if distance > max_distance:
                    continue
                silhouette = compare_silhouettes(track.signature, det.signature)
                score = 1.0 - distance / max_distance
                if silhouette is not None:
                    score = score * 0.8 + silhouette * 0.2
                if score > best_score:
                    best_id, best_score = tid, score
            if best_id is None:
                tid = self._next_id
                self._next_id += 1
                track = Track(tid, det.bbox, det.kind, det.color, det.confidence, now)
                self.tracks[tid] = track
                self.events.append({"event": "identified", "track_id": tid, "at": now})
            else:
                tid = best_id
                track = self.tracks[tid]
                dt = max(0.001, now - track.last_seen)
                ox, oy = track.bbox.center
                nx, ny = det.bbox.center
                vx, vy = (nx - ox) / dt, (ny - oy) / dt
                track.velocity = (
                    0.7 * track.velocity[0] + 0.3 * vx,
                    0.7 * track.velocity[1] + 0.3 * vy,
                )
                track.bbox = det.bbox
                track.color = det.color
                track.confidence = max(0, min(1, det.confidence))
                track.last_seen = now
                track.status = "observado"
                track.misses = 0
                if track.stable_patch is not None and det.stable_patch is not None:
                    mask = np.ones(track.stable_patch.shape[:2], np.uint8)
                    mask[:round(mask.shape[0] * 0.16)] = 0
                    change = structural_change_ratio(
                        track.stable_patch, det.stable_patch, mask
                    )
                    track.structural_change = change
                    if change is not None and change >= 0.80:
                        track.structural_streak += 1
                        if track.structural_streak >= 2:
                            track.status = "incerto"
                            self.events.append({
                                "event": "structural_change_persistent",
                                "track_id": tid, "at": now,
                                "change_ratio": round(change, 3),
                                "streak": track.structural_streak,
                            })
                    else:
                        track.structural_streak = 0
            track.signature = det.signature or track.signature
            track.stable_patch = det.stable_patch if det.stable_patch is not None else track.stable_patch
            track.fill_fraction = det.fill_fraction
            claimed.add(tid)
        for tid, track in list(self.tracks.items()):
            if tid in claimed:
                continue
            elapsed = max(0, now - track.last_seen)
            if elapsed >= self.max_absence_s and track.misses >= 3:
                self.events.append({"event": "lost", "track_id": tid, "at": now})
                del self.tracks[tid]
                continue
            # Só afirmar 'ocluído' quando uma região que deveria conter o objeto foi vista.
            predicted = track.bbox.translated(
                track.velocity[0] * min(elapsed, 0.4),
                track.velocity[1] * min(elapsed, 0.4),
                screen_size,
            )
            present = predicted is not None and any(
                r.intersection_area(predicted) >= predicted.area * 0.45
                for r in observed
            )
            track.status = "ocluido_ou_nao_detectado" if present else "nao_observado"
            if present:
                track.misses += 1
        return [track.to_dict() for track in self.tracks.values()]
