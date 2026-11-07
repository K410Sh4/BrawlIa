"""Curadoria local: amostras informativas, duplicatas e metadados auditáveis."""
from __future__ import annotations

import cv2
import numpy as np


class DatasetCurator:
    """Seleciona imagens sem confundir predições com rótulos humanos."""

    IMPORTANT_EVENTS = frozenset((
        "confirmed", "lost", "structural_change_persistent",
        "reset", "state_change", "scene_change_suspected",
    ))

    def __init__(self, interval_s: float = 0.5,
                 uncertain_interval_s: float = 0.40,
                 duplicate_hamming_max: int = 2):
        self.interval_s = interval_s
        self.uncertain_interval_s = uncertain_interval_s
        self.duplicate_hamming_max = duplicate_hamming_max
        self._last_sample = float("-inf")
        self._recent_hashes: dict[tuple, tuple[int, int, int]] = {}
        self.duplicates_skipped = 0

    def select_uncertain_frames(self, result: dict) -> bool:
        state = result.get("state", {})
        tracks = result.get("tracks", [])
        return (
            state.get("name") == "desconhecido"
            or any(t.get("status") == "incerto" for t in tracks)
        )

    def collect_failure_examples(self, result: dict) -> bool:
        return any(
            event.get("event") in self.IMPORTANT_EVENTS
            for event in result.get("events", [])
        ) or bool(result.get("scene_change_detected"))

    def select_samples(self, result: dict) -> tuple[bool, str]:
        timestamp = float(result["timestamp_s"])
        critical = self.collect_failure_examples(result)
        uncertain = self.select_uncertain_frames(result)
        if critical:
            reason = "critical_event"
        elif uncertain and (
            timestamp - self._last_sample >= self.uncertain_interval_s
        ):
            reason = "uncertain_scene"
        elif timestamp - self._last_sample >= self.interval_s:
            reason = "periodic"
        else:
            return False, "skipped"
        self._last_sample = timestamp
        return True, reason

    @staticmethod
    def _hash(frame: np.ndarray) -> tuple[int, int, int]:
        if frame.ndim != 3 or frame.size == 0:
            raise ValueError("Imagem BGR inválida")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = thumb[:, 1:] > thumb[:, :-1]
        value = 0
        for bit in bits.flat:
            value = (value << 1) | int(bit)
        return value, round(float(np.mean(gray))), round(float(np.std(gray)))

    def detect_duplicate_samples(self, frame: np.ndarray, key: tuple,
                                 *, force: bool = False) -> bool:
        """Duplicata por dHash apenas na mesma ROI, estado e resolução.

        Eventos críticos sempre são preservados; a deduplicação é consultiva.
        """
        current = self._hash(frame)
        previous = self._recent_hashes.get(key)
        self._recent_hashes[key] = current
        if len(self._recent_hashes) > 512:
            # Evita crescimento ilimitado e mantém comportamento determinístico.
            self._recent_hashes.pop(next(iter(self._recent_hashes)))
        duplicate = (
            not force
            and previous is not None
            and (current[0] ^ previous[0]).bit_count()
            <= self.duplicate_hamming_max
            and abs(current[1] - previous[1]) <= 12
            and abs(current[2] - previous[2]) <= 10
        )
        if duplicate:
            self.duplicates_skipped += 1
        return duplicate

    @staticmethod
    def validate_annotations(entry: dict) -> bool:
        """Rótulo humano aceito nunca pode vir somente da previsão automática."""
        status = entry.get("label_status", "unreviewed")
        if status in ("accepted", "corrected"):
            return bool(entry.get("human_verified")) and bool(
                entry.get("annotation")
            )
        return status in ("unreviewed", "rejected", "uncertain")

    @staticmethod
    def compare_model_versions(before: dict, after: dict) -> dict:
        """Comparativo descritivo; não inventa recall nem precisão."""
        metrics = ("effective_fps", "pixel_reduction_fraction",
                   "capture_and_analysis_p95_ms")
        return {
            key: {
                "before": before.get(key),
                "after": after.get(key),
                "difference": round(after[key] - before[key], 4)
                if isinstance(before.get(key), (float, int))
                and isinstance(after.get(key), (float, int))
                else None,
            }
            for key in metrics
        }
