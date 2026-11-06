"""PNG brutos sincronizados: fila limitada, rajadas e revisão separada."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import shutil
import threading
import time

import cv2
import numpy as np

from .dataset import proposed_boxes
from .domain import FramePacket
from .storage import new_session_id, write_json


class SampleRecorder:
    def __init__(
        self,
        parent: Path,
        sample_fps=2,
        max_samples=1500,
        max_bytes=1024**3,
        quality=86,
        *,
        session_id=None,
        burst_fps=10,
        min_free_bytes=512 * 1024**2,
    ):
        if sample_fps not in (1, 2, 5, 10, 15) or burst_fps not in (5, 10):
            raise ValueError("FPS de amostragem inválido")
        if max_samples < 1 or max_bytes < 1 or min_free_bytes < 0:
            raise ValueError("Limites de amostragem inválidos")
        self.session_id = session_id or new_session_id()
        if Path(self.session_id).name != self.session_id or self.session_id in (
            ".",
            "..",
        ):
            raise ValueError("Identificador de sessão inválido")
        self.folder = (Path(parent) / self.session_id).resolve()
        self.interval = 1 / sample_fps
        self.burst_fps = max(sample_fps, burst_fps)
        self.max_samples, self.max_bytes = max_samples, max_bytes
        self.min_free_bytes = min_free_bytes
        self.saved = self.bytes_saved = self.skipped = self.invalid = 0
        self.error = ""
        self.limit_reached = False
        self._next_sample_at = 0.0
        self._burst_until = self._cooldown_until = 0.0
        self._burst_reasons = []
        self._previous_player = None
        self._queue = deque()
        self._last_submitted = 0
        self._seq = 0
        self._cond = threading.Condition()
        self._stopping = False
        self._thread = None

    def start(self):
        self.folder.mkdir(parents=True, exist_ok=False)
        write_json(
            self.folder / "dataset_session.json",
            {
                "session_id": self.session_id,
                "format": "png",
                "coordinate_space": "raw_roi_pixels",
                "sample_fps": 1 / self.interval,
                "burst_fps": self.burst_fps,
                "max_samples": self.max_samples,
                "max_bytes": self.max_bytes,
                "note": "Previsões heurísticas não são rótulos humanos. Uma sessão deve conter uma única partida para separação de conjuntos.",
            },
        )
        self._thread = threading.Thread(
            target=self._write_loop, daemon=True, name="BrawllaSamples"
        )
        self._thread.start()

    def submit(self, frame: np.ndarray):
        """Compatibilidade para fontes externas; sessão usa observe(packet, data)."""
        self._seq += 1
        self.observe(
            FramePacket(
                self._seq,
                time.perf_counter(),
                frame,
                metadata={"session_id": self.session_id, "captured_unix": time.time()},
            ),
            {},
        )

    def _difficulty(self, data):
        reasons = list(data.get("capture", {}).get("quality", {}).get("flags", []))
        state = data.get("game_state", {})
        combat = state.get("name") == "combate"
        player = data.get("player_health_bar_candidate")
        if combat and state.get("uncertain"):
            reasons.append("uncertain_hud")
        if combat and not player:
            reasons.append("player_not_located_or_occluded")
        if player:
            if player.get("confidence", 1) < 0.65:
                reasons.append("uncertain_player_anchor")
            if player.get("health_status") == "low_fill_candidate" or (
                "health_status" not in player
                and player.get("bar_color") in ("vermelha", "amarela")
            ):
                reasons.append("possible_low_health")
            center = player.get("center")
            width = data.get("resolution", [1280])[0]
            if center and any(
                np.linalg.norm(np.array(red.get("center", [0, 0])) - center)
                < 0.03 * width
                for red in data.get(
                    "other_health_bar_candidates",
                    data.get("red_health_bar_candidates", []),
                )
            ):
                reasons.append("possible_overlap")
            if center and self._previous_player:
                width = data.get("resolution", [1280])[0]
                if (
                    np.linalg.norm(np.array(center) - self._previous_player)
                    > 0.05 * width
                ):
                    reasons.append("player_position_jump")
            self._previous_player = center
        else:
            self._previous_player = None
        return reasons

    def observe(self, packet, data):
        now = packet.timestamp_s
        reasons = self._difficulty(data)
        with self._cond:
            if (
                self._stopping
                or self.limit_reached
                or packet.sequence <= self._last_submitted
            ):
                return
            if reasons and now >= self._cooldown_until:
                self._burst_until, self._cooldown_until = now + 3, now + 10
                self._burst_reasons = reasons
                self._next_sample_at = min(self._next_sample_at, now)
            if now < self._next_sample_at:
                return
            burst = now < self._burst_until
            self._next_sample_at = now + (
                1 / self.burst_fps if burst else self.interval
            )
            if not packet.metadata.get("quality", {}).get("usable", True):
                self.invalid += 1
                return
            self._last_submitted = packet.sequence
            # Pequena fila limitada: sem duplicatas ou backlog crescente.
            if len(self._queue) >= 3:
                self._queue.popleft()
                self.skipped += 1
            self._queue.append(
                (
                    packet,
                    deepcopy(data),
                    ["burst", *self._burst_reasons] if burst else ["periodic"],
                )
            )
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stopping = True
            self._cond.notify_all()

    def join(self, timeout=5):
        if self._thread:
            self._thread.join(timeout)

    def snapshot(self):
        with self._cond:
            return {
                "folder": str(self.folder),
                "session_id": self.session_id,
                "saved": self.saved,
                "bytes_saved": self.bytes_saved,
                "skipped": self.skipped,
                "invalid": self.invalid,
                "error": self.error,
                "limit_reached": self.limit_reached,
                "format": "png",
                "queue_capacity": 3,
            }

    def _write_loop(self):
        try:
            with (self.folder / "manifest.jsonl").open(
                "a", encoding="utf-8"
            ) as manifest:
                while True:
                    with self._cond:
                        self._cond.wait_for(lambda: self._queue or self._stopping)
                        if not self._queue:
                            break
                        packet, data, reasons = self._queue.popleft()
                    frame = packet.image
                    h, w = frame.shape[:2]
                    ok, encoded = cv2.imencode(
                        ".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                    )
                    if not ok:
                        raise RuntimeError("Falha ao codificar PNG")
                    stem = f"frame_{packet.sequence:09d}"
                    entry = {
                        **packet.metadata,
                        "session_id": self.session_id,
                        "frame_id": packet.sequence,
                        "capture_timestamp_s": packet.timestamp_s,
                        "image": stem + ".png",
                        "width": w,
                        "height": h,
                        "label_status": "unreviewed",
                        "selection_reasons": reasons,
                        "prediction": stem + ".prediction.json",
                    }
                    prediction = {
                        "session_id": self.session_id,
                        "frame_id": packet.sequence,
                        "coordinate_space": "raw_roi_pixels",
                        "proposed_boxes": proposed_boxes(data, w, h),
                        "analysis": data,
                        "ground_truth": False,
                    }
                    prediction_bytes = json.dumps(
                        prediction, ensure_ascii=False
                    ).encode("utf-8")
                    line = json.dumps(entry, ensure_ascii=False) + "\n"
                    cost = (
                        encoded.nbytes
                        + len(prediction_bytes)
                        + len(line.encode("utf-8"))
                    )
                    if (
                        self.saved >= self.max_samples
                        or self.bytes_saved + cost > self.max_bytes
                        or shutil.disk_usage(self.folder).free - cost
                        < self.min_free_bytes
                    ):
                        with self._cond:
                            self.limit_reached = self._stopping = True
                            self._queue.clear()
                        break
                    image_path = self.folder / entry["image"]
                    prediction_path = self.folder / entry["prediction"]
                    try:
                        encoded.tofile(str(image_path))
                        prediction_path.write_bytes(prediction_bytes)
                        manifest.write(line)
                        manifest.flush()
                    except Exception:
                        image_path.unlink(missing_ok=True)
                        prediction_path.unlink(missing_ok=True)
                        raise
                    with self._cond:
                        self.saved += 1
                        self.bytes_saved += cost
        except Exception as exc:
            with self._cond:
                self.error = str(exc)
                self._stopping = True
                self._queue.clear()
