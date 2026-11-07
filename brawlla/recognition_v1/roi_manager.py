"""Bounded regional acquisition with semantic merging and periodic global recovery.

The planner never uses old predictions as ground truth. It prioritizes confirmed
tracks, preserves source ROIs after merging and bounds pixel/region budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import CaptureRegion, Rect
from .scene_change import scene_probe_rect


@dataclass
class CapturePlan:
    full: bool
    regions: list[CaptureRegion]
    reason: str
    coverage: float
    suppressed_targets: int = 0

    def to_dict(self) -> dict:
        return {
            "full": self.full,
            "regions": [r.to_dict() for r in self.regions],
            "reason": self.reason,
            "coverage": round(self.coverage, 4),
            "suppressed_targets": self.suppressed_targets,
        }


class ROIManager:
    def __init__(
        self,
        full_interval_s: float = 0.90,
        max_regions: int = 7,
        max_coverage: float = 0.55,
        max_targets: int = 4,
        terminal_full_interval_s: float = 1.50,
    ):
        if (
            terminal_full_interval_s <= 0
            or full_interval_s <= 0
            or max_regions < 2
            or max_targets < 1
            or not 0 < max_coverage <= 1
        ):
            raise ValueError("Politica de ROI invalida")
        self.full_interval_s = full_interval_s
        self.terminal_full_interval_s = terminal_full_interval_s
        self.max_regions = max_regions
        self.max_coverage = max_coverage
        self.max_targets = max_targets
        self.last_full: float | None = None
        self.force_reason: str | None = None
        self.full_count = 0
        self.regional_count = 0
        self.suppressed_total = 0

    def request_full_scan(self, reason: str):
        self.force_reason = reason

    def note_full(self, now: float):
        self.last_full = now
        self.force_reason = None
        self.full_count += 1

    @staticmethod
    def _merge(regions: list[CaptureRegion]) -> list[CaptureRegion]:
        """Union intersecting ROIs without losing roles or source rectangles."""
        merged: list[CaptureRegion] = []
        for region in sorted(regions, key=lambda item: item.priority, reverse=True):
            rect = region.rect
            sources = list(region.source_regions)
            reasons = [region.reason]
            priority = region.priority
            again = True
            while again:
                again = False
                for index, other in enumerate(merged):
                    if rect.intersection_area(other.rect) > 0:
                        rect = rect.merged(other.rect)
                        sources.extend(other.source_regions)
                        reasons.append(other.reason)
                        priority = max(priority, other.priority)
                        merged.pop(index)
                        again = True
                        break
            deduped = tuple(dict.fromkeys(sources))
            name = (
                deduped[0][0]
                if len(deduped) == 1
                else next(
                    (role for role, _ in deduped if role.startswith("target_")),
                    "merged",
                )
            )
            merged.append(
                CaptureRegion(
                    name,
                    rect,
                    "+".join(dict.fromkeys(reasons)),
                    priority,
                    deduped,
                )
            )
        return merged

    @staticmethod
    def _hud_regions(size: tuple[int, int]) -> list[CaptureRegion]:
        width, height = size
        # Avoid two 40%x40% quadrants: together those wasted 32% of a frame.
        top = round(height * 0.70)
        h = height - top
        side = round(width * 0.20)
        return [
            CaptureRegion("hud_left", Rect(0, top, side, h), "movement_anchor", 2),
            CaptureRegion(
                "hud_right",
                Rect(width - side, top, side, h),
                "attack_anchor",
                2,
            ),
        ]

    @staticmethod
    def _score_target(track: dict, now: float, size: tuple[int, int]) -> float:
        """Ranking is a heuristic, not a claimed probability of identity."""
        if track.get("status") in ("perdido", "candidate", "nao_observado"):
            return float("-inf")
        if not track.get("confirmed", track.get("hits", 2) >= 2):
            return float("-inf")
        # Older tracker payloads omit confidence; preserve compatible tracking.
        confidence = float(track.get("confidence", 0.55))
        if confidence < 0.48:
            return float("-inf")
        age = max(0, now - float(track.get("last_seen", now)))
        if age >= 1.2:
            return float("-inf")
        x, y, w, h = track["xywh"]
        sw, sh = size
        cx = (x + w / 2) / sw
        cy = (y + h / 2) / sh
        centrality = max(0.0, 1 - 1.2 * abs(cx - 0.5) - abs(cy - 0.5))
        green_bonus = 0.24 if track.get("color") == "verde" else 0
        player_bonus = 0.75 if track.get("kind") == "jogador_candidato" else 0
        return (
            confidence * 0.45
            + min(1, float(track.get("hits", 2)) / 5) * 0.20
            + centrality * 0.25
            + green_bonus
            + player_bonus
            - age * 0.2
        )

    @staticmethod
    def _target_region(
        track: dict, now: float, size: tuple[int, int]
    ) -> CaptureRegion | None:
        x, y, w, h = map(int, track["xywh"])
        # A track is a health-bar candidate. Include its probable body below it.
        body_w = max(round(w * 2.0), w + 32)
        body_h = max(round(w * 1.65), h + 60)
        bbox = Rect(
            round(x + w / 2 - body_w / 2),
            round(y - w * 0.20),
            body_w,
            body_h,
        )
        dt = max(0, now - float(track["last_seen"]))
        vx, vy = track.get("velocity", [0.0, 0.0])
        predicted = bbox.translated(vx * min(dt, 0.35), vy * min(dt, 0.35), size)
        if predicted is None:
            return None
        margin = 0.45 if track.get("status") != "observado" else 0.28
        rect = predicted.expand(margin, size)
        if rect is None:
            return None
        tid = track["track_id"]
        return CaptureRegion("target_" + str(tid), rect, "confirmed_track", 3)

    @staticmethod
    def _coverage(regions: list[CaptureRegion], size: tuple[int, int]) -> float:
        return sum(r.rect.area for r in regions) / (size[0] * size[1])

    def plan(
        self,
        now: float,
        screen_size: tuple[int, int],
        state: str,
        tracks: list[dict],
    ) -> CapturePlan:
        width, height = screen_size
        if width < 320 or height < 200:
            raise ValueError("Dimensao invalida")
        interval = (
            self.terminal_full_interval_s
            if state in ("derrota", "resultados")
            else self.full_interval_s
        )
        if (
            self.force_reason
            or self.last_full is None
            or now - self.last_full >= interval
        ):
            return CapturePlan(
                True, [], self.force_reason or "varredura_periodica", 1.0
            )

        if state in ("derrota", "resultados"):
            # Resultado/derrota usam uma sonda de cena, não os botões do jogo.
            probe = CaptureRegion(
                "scene_probe", scene_probe_rect(screen_size),
                "scene_transition", 2,
            )
            self.regional_count += 1
            return CapturePlan(
                False, [probe], "estado_terminal_regional",
                self._coverage([probe], screen_size),
            )
        regions = self._hud_regions(screen_size)
        if state != "combate":
            merged = self._merge(regions)
            self.regional_count += 1
            return CapturePlan(
                False, merged, "estado_estavel_regional",
                self._coverage(merged, screen_size),
            )

        ranked = sorted(
            tracks,
            key=lambda t: self._score_target(t, now, screen_size),
            reverse=True,
        )
        eligible = [
            t
            for t in ranked
            if self._score_target(t, now, screen_size) != float("-inf")
        ]
        suppressed = max(0, len(eligible) - self.max_targets)
        accepted = 0
        for track in eligible[: self.max_targets]:
            candidate = self._target_region(track, now, screen_size)
            if candidate is None:
                suppressed += 1
                continue
            trial = self._merge(regions + [candidate])
            if (
                len(trial) <= self.max_regions
                and self._coverage(trial, screen_size) <= self.max_coverage
            ):
                regions.append(candidate)
                accepted += 1
            else:
                suppressed += 1
        if accepted == 0:
            # Cheap central search prevents total blindness while tracks warm up.
            center = Rect(
                round(width * 0.37),
                round(height * 0.31),
                round(width * 0.26),
                round(height * 0.35),
            )
            trial = self._merge(
                regions + [CaptureRegion("search_center", center, "discovery", 1)]
            )
            if self._coverage(trial, screen_size) <= self.max_coverage:
                regions.append(CaptureRegion("search_center", center, "discovery", 1))
        merged = self._merge(regions)
        self.regional_count += 1
        self.suppressed_total += suppressed
        reason = "captura_estrategica" if not suppressed else "alvos_limitados"
        return CapturePlan(
            False,
            merged,
            reason,
            self._coverage(merged, screen_size),
            suppressed,
        )
