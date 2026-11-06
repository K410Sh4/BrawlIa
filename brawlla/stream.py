"""Pipeline de captura e análise com backpressure: nunca processar frames antigos."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable

import cv2
import numpy as np
from .domain import FramePacket
from .adaptive import AdaptiveFps
from .config import FPS_CHOICES
from .dataset import QualityMonitor, crop_frame


class LatestFrame:
    """Armazena só um quadro recente; produtor não espera pelo consumidor."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._seq = 0
        self._stopped = False
        self._packet = None

    def publish(
        self,
        frame: np.ndarray,
        capture_ms: float = 0.0,
        captured_at: float | None = None,
        metadata: dict | None = None,
    ) -> int:
        with self._condition:
            timestamp = time.perf_counter() if captured_at is None else captured_at
            packet = FramePacket(
                self._seq + 1, timestamp, frame, capture_ms, metadata or {}
            )
            self._seq, self._packet = packet.sequence, packet
            self._condition.notify_all()
            return self._seq

    def take_packet_after(self, last_seq: int, timeout: float = 0.25):
        packet = self.take_latest_after(last_seq, timeout)
        return (
            (packet.sequence, packet.image, packet.timestamp_s, packet.capture_ms)
            if packet
            else (last_seq, None, 0.0, 0.0)
        )

    def take_latest_after(self, last_seq: int, timeout=0.25):
        with self._condition:
            self._condition.wait_for(
                lambda: self._seq > last_seq or self._stopped, timeout=timeout
            )
            return self._packet if self._seq > last_seq else None

    def take_after(self, last_seq: int, timeout: float = 0.25):
        seq, frame, _, _ = self.take_packet_after(last_seq, timeout)
        return seq, frame

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


class RateCounter:
    """Média móvel em janela de 2 segundos para FPS real, não FPS solicitado."""

    def __init__(self, window_s: float = 2.0):
        self.window_s = window_s
        self.events = deque()
        self._lock = threading.Lock()

    def tick(self) -> None:
        now = time.perf_counter()
        with self._lock:
            self.events.append(now)
            self._prune(now)

    def _prune(self, now: float) -> None:
        while self.events and now - self.events[0] > self.window_s:
            self.events.popleft()

    def fps(self) -> float:
        with self._lock:
            now = time.perf_counter()
            self._prune(now)
            if len(self.events) < 2:
                return 0.0
            duration = max(now - self.events[0], 0.001)
            return max(0.0, (len(self.events) - 1) / duration)


def resize_for_analysis(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    if max_width < 320:
        raise ValueError("Largura da análise deve ser de pelo menos 320.")
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    s = max_width / w
    return cv2.resize(
        frame, (max_width, max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )


@dataclass
class PipelineStats:
    captured: int = 0
    analyzed: int = 0
    skipped: int = 0
    error: str = ""
    last_latency_ms: float = 0.0
    capture_ms: float = 0.0
    frame_age_ms: float = 0.0


class VisionPipeline:
    """2 workers: captura 15/30/60 FPS; inferência no frame mais recente.

    A interface Tk deve consumir resultados em `on_result` (executado em thread
    de trabalho: NÃO chamar widgets Tk ali). `on_frame` recebe frames crus no
    thread de captura e deve ser não bloqueante.
    """

    def __init__(
        self,
        source_factory: Callable[[], object],
        analyzer: Callable,
        on_result: Callable,
        on_frame: Callable | None = None,
        target_fps: int = 30,
        max_width: int = 1280,
        analysis_fps: int = 15,
        *,
        timestamped_analyzer: bool = False,
        on_observation: Callable | None = None,
        session_id: str = "",
        roi=None,
        source_aspect: float = 0,
        auto_fps: bool = False,
    ):
        if target_fps not in FPS_CHOICES:
            raise ValueError("FPS de captura deve ser 5, 10, 15, 30 ou 60.")
        if analysis_fps not in FPS_CHOICES:
            raise ValueError("FPS de análise deve ser 5, 10, 15, 30 ou 60.")
        if max_width < 320:
            raise ValueError("Largura da análise deve ser de pelo menos 320.")
        self.on_observation = on_observation
        self.session_id = session_id
        self.roi, self.source_aspect = roi, source_aspect
        self.quality_monitor = QualityMonitor()
        self.timestamped_analyzer = timestamped_analyzer
        self.source_factory = source_factory
        self.analyzer = analyzer
        self.on_result = on_result
        self.on_frame = on_frame
        self.target_fps = target_fps
        self.analysis_fps = min(analysis_fps, target_fps)
        self.auto_fps = auto_fps
        self.adaptive = AdaptiveFps(target_fps, self.analysis_fps) if auto_fps else None
        self.max_width = max_width
        self.stats = PipelineStats()
        self.slot = LatestFrame()
        self.capture_rate = RateCounter()
        self.analysis_rate = RateCounter()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stats_lock = threading.Lock()
        self._latencies = deque(maxlen=120)
        self._started_at = None
        self._last_capture_at = None
        self._last_result_at = None
        self._quality = {}

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("Pipeline já iniciado")
        self._started_at = time.perf_counter()
        self._threads = [
            threading.Thread(
                target=self._capture_loop, daemon=True, name="BrawllaCapture"
            ),
            threading.Thread(
                target=self._analysis_loop, daemon=True, name="BrawllaAnalysis"
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.slot.stop()

    def join(self, timeout: float | None = None) -> None:
        for thread in self._threads:
            if thread.ident is not None:
                thread.join(timeout=timeout)

    def snapshot(self) -> dict:
        with self._stats_lock:
            counts = vars(self.stats).copy()
            from .metrics import percentile95

            counts["analysis_p95_ms"] = percentile95(self._latencies)
            counts["adaptive_fps"] = (
                self.adaptive.snapshot()
                if self.adaptive is not None
                else {"enabled": False}
            )
            now = time.perf_counter()
            capture_reference = (
                self._last_capture_at
                if self._last_capture_at is not None
                else self._started_at
            )
            result_reference = (
                self._last_result_at
                if self._last_result_at is not None
                else self._started_at
            )
            capture_age = now - (
                capture_reference if capture_reference is not None else now
            )
            result_age = now - (
                result_reference if result_reference is not None else now
            )
            counts["quality"] = self._quality
            counts["replaced_before_analysis"] = self.stats.skipped
            counts["capture_stalled"] = bool(
                self._started_at is not None
                and not self._stop.is_set()
                and capture_age > 2
            )
            counts["analysis_stalled"] = bool(
                self._started_at is not None
                and not self._stop.is_set()
                and result_age > 2
                and capture_age <= 2
            )
            counts["seconds_since_capture"] = round(capture_age, 3)
            counts["seconds_since_result"] = round(result_age, 3)
        counts["capture_fps"] = round(self.capture_rate.fps(), 1)
        counts["analysis_fps"] = round(self.analysis_rate.fps(), 1)
        return counts

    def _set_error(self, exc: Exception) -> None:
        with self._stats_lock:
            self.stats.error = str(exc)
        self.stop()

    def _capture_loop(self) -> None:
        source = None
        try:
            source = self.source_factory()  # MSS no próprio thread
            next_tick = time.perf_counter()
            while not self._stop.is_set():
                capture_started = time.perf_counter()
                captured_unix = time.time()
                frame = source.grab()
                capture_ms = (time.perf_counter() - capture_started) * 1000
                if self.adaptive is not None:
                    self.adaptive.record_capture(capture_ms)
                if frame is None:
                    raise RuntimeError("Capturador retornou um quadro vazio")
                with self._stats_lock:
                    self._last_capture_at = time.perf_counter()
                    self.stats.captured += 1
                    self.stats.capture_ms = capture_ms
                frame, geometry = crop_frame(frame, self.roi, self.source_aspect)
                quality = self.quality_monitor.inspect(frame, capture_started)
                quality["flags"].extend(getattr(source, "quality_flags", []))
                if isinstance(getattr(source, "quality_flags", None), list):
                    quality["occlusion_detection"] = "sampled_window_hit_test_heuristic"
                with self._stats_lock:
                    self._quality = quality
                self.slot.publish(
                    frame,
                    capture_ms,
                    capture_started,
                    {
                        "session_id": self.session_id,
                        "captured_unix": captured_unix,
                        **geometry,
                        "quality": quality,
                        "source_type": type(source).__name__,
                        "source_label": getattr(source, "source_label", ""),
                        "window_rect_screen_xywh": getattr(source, "_last_rect", None),
                    },
                )
                self.capture_rate.tick()
                if self.on_frame is not None:
                    try:
                        self.on_frame(frame)
                    except Exception as exc:
                        self._set_error(exc)
                        break
                period = 1.0 / self.target_fps
                next_tick += period
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    self._stop.wait(sleep_s)
                else:
                    next_tick = time.perf_counter()  # Sem backlog artificial.
        except Exception as exc:
            self._set_error(exc)
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass
            self.slot.stop()

    def _analysis_loop(self) -> None:
        seq = 0
        try:
            while not self._stop.is_set():
                packet = self.slot.take_latest_after(seq)
                if packet is None:
                    continue
                raw, captured_at, capture_ms = (
                    packet.image,
                    packet.timestamp_s,
                    packet.capture_ms,
                )
                skipped = max(0, packet.sequence - seq - 1)
                seq = packet.sequence
                started = time.perf_counter()
                image = resize_for_analysis(raw, self.max_width)
                resize_ms = (time.perf_counter() - started) * 1000
                annotated, detections = (
                    self.analyzer(image, timestamp_s=captured_at)
                    if self.timestamped_analyzer
                    else self.analyzer(image)
                )
                latency_ms = (time.perf_counter() - started) * 1000
                age_ms = (time.perf_counter() - captured_at) * 1000
                detections = {
                    **detections,
                    "session_id": self.session_id,
                    "frame_id": packet.sequence,
                    "capture_timestamp_s": captured_at,
                    "capture": {
                        **packet.metadata,
                        "analysis_resolution": [image.shape[1], image.shape[0]],
                        "raw_roi_resolution": [raw.shape[1], raw.shape[0]],
                        "raw_per_analysis_xy": [
                            raw.shape[1] / image.shape[1],
                            raw.shape[0] / image.shape[0],
                        ],
                    },
                    "pipeline_timings_ms": {
                        "capture": round(capture_ms, 3),
                        "resize": round(resize_ms, 3),
                        "analysis_with_resize": round(latency_ms, 3),
                        "frame_age_at_result": round(age_ms, 3),
                    },
                }
                with self._stats_lock:
                    self._latencies.append(latency_ms)
                    self.stats.frame_age_ms = age_ms
                    self._last_result_at = time.perf_counter()
                    self.stats.analyzed += 1
                    self.stats.skipped += skipped
                    self.stats.last_latency_ms = latency_ms
                with self._stats_lock:
                    detections["processing_counters"] = {
                        "captured": self.stats.captured,
                        "analyzed": self.stats.analyzed,
                        "replaced_before_analysis": self.stats.skipped,
                        "capture_target_fps": self.target_fps,
                        "analysis_target_fps": self.analysis_fps,
                    }
                self.analysis_rate.tick()
                if self.on_observation:
                    self.on_observation(packet, detections)
                self.on_result(annotated, detections)
                if self.adaptive is not None:
                    self.adaptive.record_analysis(
                        (time.perf_counter() - started) * 1000, age_ms
                    )
                    self.target_fps, self.analysis_fps = self.adaptive.rates()
                period = 1.0 / self.analysis_fps
                remaining = period - (time.perf_counter() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except Exception as exc:
            self._set_error(exc)
