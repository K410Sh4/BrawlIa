"""Reconhecimento contextual V1: observação local, sem controle do jogo."""

from .geometry import Rect, CaptureRegion
from .hud_context import HUDContext, HUDAnchors
from .context_engine import ContextMemory, Detection
from .roi_manager import ROIManager

__all__ = [
    "Rect",
    "CaptureRegion",
    "HUDContext",
    "HUDAnchors",
    "ContextMemory",
    "Detection",
    "ROIManager",
]
