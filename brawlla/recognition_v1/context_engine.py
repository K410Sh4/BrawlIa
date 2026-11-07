"""Contextual identity with candidate confirmation, TTL and explicit missing evidence.

A visual candidate is not a confirmed participant. Unobserved regions never
count as empty. Structural change flags uncertainty; it does not destroy IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .geometry import Rect
from .silhouette import Signature, compare_silhouettes, structural_change_ratio


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
    body_support: float = 0.0


@dataclass
class Track:
    track_id: int
    bbox: Rect
    kind: str
    color: str
    confidence: float
    last_seen: float
    velocity: tuple[float, float] = (0.0, 0.0)
    status: str = "candidate"
    hits: int = 1
    misses: int = 0
    signature: Signature | None = field(default=None, repr=False)
    fill_fraction: float | None = None
    stable_patch: np.ndarray | None = field(default=None, repr=False)
    structural_change: float | None = None
    structural_streak: int = 0
    body_support: float = 0.0

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
            "hits": self.hits,
            "confirmed": self.hits >= 2,
            "misses": self.misses,
            "fill_fraction": self.fill_fraction,
            "body_support": round(self.body_support, 3),
            "health_exact": None,
            "structural_change": self.structural_change,
            "structural_streak": self.structural_streak,
        }


class ContextMemory:
    def __init__(
        self,
        max_absence_s: float = 2.5,
        max_speed_px_s: float = 650,
        candidate_absence_s: float = 0.45,
        max_tracks: int = 36,
        max_new_per_frame: int = 10,
    ):
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.max_absence_s = max_absence_s
        self.candidate_absence_s = candidate_absence_s
        self.max_speed_px_s = max_speed_px_s
        self.max_tracks = max_tracks
        self.max_new_per_frame = max_new_per_frame
        self.events: list[dict] = []

    def reset(self, reason: str):
        self.events.append({"event": "reset", "reason": reason})
        self.tracks.clear()

    def _drop_expired(self, now: float):
        for tid, track in list(self.tracks.items()):
            timeout = (
                self.max_absence_s if track.hits >= 2 else self.candidate_absence_s
            )
            if now - track.last_seen >= timeout:
                self.events.append(
                    {
                        "event": "lost" if track.hits >= 2 else "candidate_expired",
                        "track_id": tid,
                        "at": now,
                    }
                )
                del self.tracks[tid]

    def _match(self, det: Detection, used: set[int], now: float) -> Track | None:
        best, best_score = None, -1.0
        for track in self.tracks.values():
            if track.track_id in used or track.kind != det.kind:
                continue
            elapsed = max(0.001, now - track.last_seen)
            cx, cy = track.bbox.center
            px = cx + track.velocity[0] * min(elapsed, 0.35)
            py = cy + track.velocity[1] * min(elapsed, 0.35)
            dx = det.bbox.center[0] - px
            dy = det.bbox.center[1] - py
            distance = math.hypot(dx, dy)
            max_distance = max(35, min(200, self.max_speed_px_s * elapsed + 30))
            if distance > max_distance:
                continue
            silhouette = compare_silhouettes(track.signature, det.signature)
            similarity = 1.0 - distance / max_distance
            if silhouette is not None:
                similarity = 0.82 * similarity + 0.18 * silhouette
            if track.color == det.color:
                similarity += 0.04
            if similarity > best_score:
                best, best_score = track, similarity
        return best if best_score >= 0.18 else None

    def _update_track(self, track: Track, det: Detection, now: float):
        elapsed = max(0.001, now - track.last_seen)
        ox, oy = track.bbox.center
        nx, ny = det.bbox.center
        vx, vy = (nx - ox) / elapsed, (ny - oy) / elapsed
        track.velocity = (
            0.7 * track.velocity[0] + 0.3 * vx,
            0.7 * track.velocity[1] + 0.3 * vy,
        )
        if track.stable_patch is not None and det.stable_patch is not None:
            mask = np.ones(track.stable_patch.shape[:2], np.uint8)
            mask[: round(mask.shape[0] * 0.16)] = 0
            change = structural_change_ratio(track.stable_patch, det.stable_patch, mask)
            track.structural_change = change
            if change is not None and change >= 0.80:
                track.structural_streak += 1
                if track.structural_streak >= 2:
                    self.events.append(
                        {
                            "event": "structural_change_persistent",
                            "track_id": track.track_id,
                            "at": now,
                            "change_ratio": round(change, 3),
                            "streak": track.structural_streak,
                        }
                    )
            else:
                track.structural_streak = 0
        track.bbox = det.bbox
        track.color = det.color
        track.confidence = max(0.0, min(1.0, det.confidence))
        track.last_seen = now
        track.hits += 1
        track.misses = 0
        track.signature = det.signature or track.signature
        track.body_support = det.body_support
        if det.stable_patch is not None:
            track.stable_patch = det.stable_patch
        track.fill_fraction = det.fill_fraction
        track.status = "incerto" if track.structural_streak >= 2 else "observado"
        if track.hits == 2:
            self.events.append(
                {
                    "event": "confirmed",
                    "track_id": track.track_id,
                    "at": now,
                }
            )

    def update(
        self,
        detections: list[Detection],
        observed: list[Rect],
        now: float,
        screen_size: tuple[int, int],
    ) -> list[dict]:
        """Confirm after two distinct observations, never infer from uncaptured pixels."""
        self._drop_expired(now)
        claimed: set[int] = set()
        new_count = 0
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if not 0.48 <= det.confidence <= 1.0:
                continue
            matched = self._match(det, claimed, now)
            if matched is not None:
                self._update_track(matched, det, now)
                claimed.add(matched.track_id)
            elif (
                len(self.tracks) < self.max_tracks
                and new_count < self.max_new_per_frame
            ):
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(
                    tid,
                    det.bbox,
                    det.kind,
                    det.color,
                    det.confidence,
                    now,
                    signature=det.signature,
                    stable_patch=det.stable_patch,
                    fill_fraction=det.fill_fraction,
                    body_support=det.body_support,
                )
                self.events.append(
                    {
                        "event": "candidate",
                        "track_id": tid,
                        "at": now,
                    }
                )
                claimed.add(tid)
                new_count += 1
        for tid, track in self.tracks.items():
            if tid in claimed:
                continue
            elapsed = max(0, now - track.last_seen)
            predicted = track.bbox.translated(
                track.velocity[0] * min(elapsed, 0.35),
                track.velocity[1] * min(elapsed, 0.35),
                screen_size,
            )
            was_observed = predicted is not None and any(
                rect.intersection_area(predicted) >= predicted.area * 0.45
                for rect in observed
            )
            track.status = (
                "ocluido_ou_nao_detectado" if was_observed else "nao_observado"
            )
            if was_observed:
                track.misses += 1
        return [track.to_dict() for track in self.tracks.values()]
