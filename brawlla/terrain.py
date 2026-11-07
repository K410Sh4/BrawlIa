"""Amostragem temporal do terreno, separada das barras e do jogador.

Contornos de vegetação e paredes são evidências auxiliares de cenário,
não evidências de identidade. Reutilização é marcada como aproximada.
"""

import time

import cv2
import numpy as np

from .vision import Region, _terrain


class TerrainObserver:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.scene = None
        self.resolution = None
        self.last_refresh_at = None
        self.frames_since_refresh = 0
        self.refreshes = 0
        self.reuses = 0

    def _refresh(self, hsv):
        height, width = hsv.shape[:2]
        terrain_width = min(width, self.config.terrain_width)
        terrain_height = max(1, round(height * terrain_width / width))
        small = (
            cv2.resize(
                hsv,
                (terrain_width, terrain_height),
                interpolation=cv2.INTER_NEAREST,
            )
            if width > terrain_width
            else hsv
        )
        sx, sy = width / terrain_width, height / terrain_height
        affine = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float32)
        result = []
        for kind in ("bush", "wall"):
            contours, _ = _terrain(small, kind, self.config, include_contours=True)
            scaled = [
                np.rint(cv2.transform(c.astype(np.float32), affine)).astype(np.int32)
                for c in contours
            ]
            result.append((kind, scaled))
        return result

    @staticmethod
    def _camera_affine(camera_motion):
        if not isinstance(camera_motion, dict) or not camera_motion.get("valid"):
            return None
        matrix = np.asarray(camera_motion.get("affine_original", []), dtype=np.float32)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            return None
        return matrix

    def observe(self, hsv, now=None, camera_motion=None):
        now = time.perf_counter() if now is None else float(now)
        size = (hsv.shape[1], hsv.shape[0])
        if size != self.resolution or self.last_refresh_at is None:
            self.reset()
            self.resolution = size
        age = now - self.last_refresh_at if self.last_refresh_at is not None else 0
        refresh = (
            self.scene is None
            or age < 0
            or age >= self.config.terrain_cache_max_s
            or self.frames_since_refresh >= self.config.terrain_sample_every - 1
        )
        if refresh:
            self.scene = self._refresh(hsv)
            self.last_refresh_at = now
            self.frames_since_refresh = 0
            self.refreshes += 1
            applied_motion = False
        else:
            self.frames_since_refresh += 1
            self.reuses += 1
            affine = self._camera_affine(camera_motion)
            applied_motion = affine is not None
            if affine is not None:
                self.scene = [
                    (
                        kind,
                        [
                            np.rint(cv2.transform(c.astype(np.float32), affine)).astype(
                                np.int32
                            )
                            for c in contours
                        ],
                    )
                    for kind, contours in self.scene
                ]
        regions = {
            kind: [
                Region(
                    *cv2.boundingRect(c),
                    round(float(cv2.contourArea(c))),
                )
                for c in contours
            ]
            for kind, contours in self.scene
        }
        return (
            regions,
            self.scene,
            {
                "refreshed": refresh,
                "cache_age_ms": round(max(0, (now - self.last_refresh_at) * 1000), 2),
                "motion_compensated": applied_motion,
                "approximate": not refresh,
                "refreshes": self.refreshes,
                "reuses": self.reuses,
            },
        )
