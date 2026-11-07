"""Coordenadas inteiras sempre relativas à imagem INTEIRA da janela."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self):
        if not all(type(v) is int for v in (self.x, self.y, self.w, self.h)):
            raise TypeError("Rect exige coordenadas inteiras")
        if self.w <= 0 or self.h <= 0:
            raise ValueError("Rect deve ter largura e altura positivas")

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def xywh(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def clipped(self, size: tuple[int, int]) -> Optional["Rect"]:
        width, height = size
        left = max(0, self.x)
        top = max(0, self.y)
        right = min(width, self.x + self.w)
        bottom = min(height, self.y + self.h)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def expand(self, fraction: float, size: tuple[int, int]) -> Optional["Rect"]:
        if not 0 <= fraction <= 4:
            raise ValueError("Margem inválida")
        mx, my = round(self.w * fraction), round(self.h * fraction)
        return Rect(self.x - mx, self.y - my, self.w + 2 * mx, self.h + 2 * my).clipped(
            size
        )

    def translated(
        self, dx: float, dy: float, size: tuple[int, int]
    ) -> Optional["Rect"]:
        return Rect(round(self.x + dx), round(self.y + dy), self.w, self.h).clipped(
            size
        )

    def intersection_area(self, other: "Rect") -> int:
        w = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        h = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        return w * h

    def overlap(self, other: "Rect") -> float:
        inter = self.intersection_area(other)
        return inter / max(1, self.area + other.area - inter)

    def merged(self, other: "Rect") -> "Rect":
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        right = max(self.x + self.w, other.x + other.w)
        bottom = max(self.y + self.h, other.y + other.h)
        return Rect(x, y, right - x, bottom - y)


@dataclass(frozen=True)
class CaptureRegion:
    name: str
    rect: Rect
    reason: str
    priority: int = 1
    # Each constituent region survives merging; coordinates remain window-relative.
    sources: tuple[tuple[str, Rect], ...] = ()

    @property
    def source_regions(self) -> tuple[tuple[str, Rect], ...]:
        return self.sources or ((self.name, self.rect),)

    @property
    def has_target(self) -> bool:
        return any(role.startswith("target_") or role == "search_center"
                   for role, _ in self.source_regions)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "xywh": self.rect.xywh(),
            "reason": self.reason,
            "priority": self.priority,
            "sources": [
                {"role": role, "xywh": region.xywh()}
                for role, region in self.source_regions
            ],
        }
