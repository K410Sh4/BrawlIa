"""30 Hz producer/consumer with a single latest-frame slot and bounded latency.

No OCR, no HSV, no video encoder, no neural network and no game commands.
30 FPS is a measured throughput target, not a claim about hardware.
"""

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
import zlib


TARGET_FPS = 30
PERIOD_NS = round(1_000_000_000 / TARGET_FPS)


@dataclass(frozen=True, slots=True)
class Frame:
    number: int
    timestamp_ns: int
    pixels: object


class LatestFrame:
    """Bounded handoff: a slow analyzer never builds a queue of stale frames."""

    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.version = 0
        self.read_version = 0
        self.replacements = 0
        self.closed = False

    def put(self, frame):
        with self.condition:
            if self.closed:
                return False
            if self.version > self.read_version:
                self.replacements += 1
            self.frame = frame
            self.version += 1
            self.condition.notify()
            return True

    def take(self, timeout=0.5):
        with self.condition:
            self.condition.wait_for(lambda: self.version > self.read_version or self.closed, timeout)
            if self.version <= self.read_version:
                return None
            self.read_version = self.version
            return self.frame

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class Metrics:
    """Measurements of *actual* capture and processing, not video container FPS."""

    def __init__(self, target_fps=TARGET_FPS):
        self.lock = threading.Lock()
        self.target_fps = target_fps
        self.started_at = time.perf_counter_ns()
        self.capture_count = 0
        self.process_count = 0
        self.missed_deadlines = 0
        self.content_checked = 0
        self.content_unchanged = 0
        self.capture_stamps = deque(maxlen=180)
        self.process_stamps = deque(maxlen=180)
        self.capture_cost_ms = deque(maxlen=180)
        self.process_cost_ms = deque(maxlen=180)
        self.frame_age_ms = deque(maxlen=180)
        self.error = None

    def captured(self, stamp, cost_ms, missed):
        with self.lock:
            self.capture_count += 1
            self.missed_deadlines += missed
            self.capture_stamps.append(stamp)
            self.capture_cost_ms.append(cost_ms)

    def processed(self, stamp, cost_ms, age_ms):
        with self.lock:
            self.process_count += 1
            self.process_stamps.append(stamp)
            self.process_cost_ms.append(cost_ms)
            self.frame_age_ms.append(age_ms)

    def visual_probe(self, unchanged):
        with self.lock:
            self.content_checked += 1
            if unchanged:
                self.content_unchanged += 1

    def failed(self, exc):
        with self.lock:
            self.error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _fps(samples):
        if len(samples) < 2 or samples[-1] <= samples[0]:
            return 0.0
        return (len(samples) - 1) * 1_000_000_000 / (samples[-1] - samples[0])

    @staticmethod
    def _p95(samples):
        if not samples:
            return 0.0
        ordered = sorted(samples)
        return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]

    def snapshot(self, *, replaced=0):
        with self.lock:
            elapsed = max((time.perf_counter_ns() - self.started_at) / 1e9, 0.001)
            return {
                "target_fps": self.target_fps,
                "elapsed_s": round(elapsed, 3),
                "capture_count": self.capture_count,
                "process_count": self.process_count,
                "capture_average_fps": round(self.capture_count / elapsed, 2),
                "process_average_fps": round(self.process_count / elapsed, 2),
                "capture_recent_fps": round(self._fps(self.capture_stamps), 2),
                "process_recent_fps": round(self._fps(self.process_stamps), 2),
                "capture_p95_ms": round(self._p95(self.capture_cost_ms), 2),
                "process_p95_ms": round(self._p95(self.process_cost_ms), 2),
                "frame_age_p95_ms": round(self._p95(self.frame_age_ms), 2),
                "replaced_before_processing": replaced,
                "missed_capture_deadlines": self.missed_deadlines,
                "visual_samples_checked": self.content_checked,
                "visually_unchanged_samples": self.content_unchanged,
                "visual_change_note": "Amostras iguais podem indicar tela parada, não necessariamente frame duplicado",
                "error": self.error,
                "video_recorded": False,
                "png_recorded": False,
                "neural_inference_enabled": False,
            }


def verify_frame(frame):
    """Minimal integrity check; an actual vision model comes in Phase 2."""
    pixels = frame.pixels
    if getattr(pixels, "ndim", 0) != 3 or pixels.shape[2] != 4:
        raise ValueError("Quadro BGRA inválido")
    # Real pixel access makes a disconnected/fake frame counter less likely.
    _ = int(pixels[pixels.shape[0] // 2, pixels.shape[1] // 2, 0])


class Pipeline:
    def __init__(self, source_factory, process=verify_frame):
        self.source_factory = source_factory
        self.process = process
        self.frames = LatestFrame()
        self.metrics = Metrics()
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.process_thread = None
        self._last_probe = None

    def start(self):
        if self.capture_thread is not None:
            raise RuntimeError("Pipeline já iniciado")
        self.capture_thread = threading.Thread(target=self._capture_loop, name="V2Capture", daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, name="V2Vision", daemon=True)
        self.process_thread.start()
        self.capture_thread.start()
        return self

    def _fail(self, error):
        self.metrics.failed(error)
        self.stop_event.set()
        self.frames.close()

    def _capture_loop(self):
        source = None
        try:
            source = self.source_factory()
            source.open()
            deadline = time.perf_counter_ns()
            index = 0
            while not self.stop_event.is_set():
                now = time.perf_counter_ns()
                if now < deadline and self.stop_event.wait((deadline - now) / 1e9):
                    break
                began = time.perf_counter_ns()
                pixels = source.grab()
                captured = time.perf_counter_ns()
                index += 1
                if not self.frames.put(Frame(index, captured, pixels)):
                    break
                deadline += PERIOD_NS
                missed = 0
                if captured >= deadline:
                    missed = (captured - deadline) // PERIOD_NS + 1
                    deadline += missed * PERIOD_NS
                self.metrics.captured(captured, (captured - began) / 1e6, missed)
        except Exception as exc:
            self._fail(exc)
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    self.metrics.failed(exc)
            self.frames.close()

    def _process_loop(self):
        try:
            while not self.stop_event.is_set():
                frame = self.frames.take(timeout=0.25)
                if frame is None:
                    if self.frames.closed:
                        break
                    continue
                started = time.perf_counter_ns()
                self.process(frame)
                # Lightweight pixel-change diagnostic. This checks actual content,
                # not just that the 30 Hz scheduler woke up. A static menu will
                # legitimately produce identical samples across many frames.
                pix = frame.pixels
                signature = (pix.shape, zlib.crc32(pix[::32, ::32, :3].tobytes()))
                self.metrics.visual_probe(signature == self._last_probe)
                self._last_probe = signature
                done = time.perf_counter_ns()
                self.metrics.processed(done, (done - started) / 1e6, (done - frame.timestamp_ns) / 1e6)
        except Exception as exc:
            self._fail(exc)

    def snapshot(self):
        return self.metrics.snapshot(replaced=self.frames.replacements)

    def stop(self):
        self.stop_event.set()
        self.frames.close()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=3)
        if self.process_thread is not None:
            self.process_thread.join(timeout=3)
        return self.snapshot()
