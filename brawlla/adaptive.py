"""Controle adaptativo de FPS por capacidade observada, sem sondar GPUs.

A capacidade efetiva (OpenCV CPU ou ONNX com GPU, quando usado) é inferida
pela latência de cada etapa. Não há afirmação de monitorar utilização de GPU.
"""

from collections import deque
import math
import threading
import time


RATE_STEPS = (5, 8, 10, 12, 15, 20, 24, 30, 40, 45, 60)


def percentile(values, quantile):
    """Percentil nearest-rank determinístico, sem dependências externas."""
    if not values:
        return 0.0
    values = sorted(values)
    return float(values[max(0, math.ceil(len(values) * quantile) - 1)])


def stepped_limit(limit, floor=5):
    """Maior FPS permitido pela capacidade observada."""
    return max(floor, max((r for r in RATE_STEPS if r <= limit), default=floor))


def next_step(current, direction):
    index = RATE_STEPS.index(current)
    index = max(0, min(len(RATE_STEPS) - 1, index + direction))
    return RATE_STEPS[index]


class AdaptiveFps:
    """Define as taxas para o próximo ciclo, com desaceleração e recuperação gradual.

    Chame record_capture no produtor e record_analysis no consumidor. Nenhuma
    espera ativa; a integração aplica as taxas apenas nos respectivos workers.
    """

    def __init__(
        self,
        capture_fps=30,
        analysis_fps=15,
        *,
        max_fps=60,
        adjust_interval_s=2.0,
        min_samples=8,
    ):
        if (
            capture_fps not in RATE_STEPS
            or analysis_fps not in RATE_STEPS
            or analysis_fps > capture_fps
            or max_fps not in RATE_STEPS
            or adjust_interval_s <= 0
            or min_samples < 1
        ):
            raise ValueError("Parâmetros de controle adaptativo inválidos")
        self._lock = threading.Lock()
        self.capture_fps = capture_fps
        self.analysis_fps = analysis_fps
        self.max_fps = max_fps
        self.interval = adjust_interval_s
        self.min_samples = min_samples
        self.capture_cost_ms = deque(maxlen=120)
        self.analysis_cost_ms = deque(maxlen=120)
        self.frame_age_ms = deque(maxlen=120)
        self._last_adjust = None
        self._good_windows = 0
        self.adjustments = 0
        self.reason = "calibrando"

    def record_capture(self, elapsed_ms):
        if math.isfinite(elapsed_ms) and elapsed_ms > 0:
            with self._lock:
                self.capture_cost_ms.append(float(elapsed_ms))

    def record_analysis(self, elapsed_ms, age_ms, *, now=None):
        if (
            not math.isfinite(elapsed_ms)
            or elapsed_ms <= 0
            or not math.isfinite(age_ms)
            or age_ms < 0
        ):
            return
        now = time.perf_counter() if now is None else float(now)
        with self._lock:
            self.analysis_cost_ms.append(float(elapsed_ms))
            self.frame_age_ms.append(float(age_ms))
            if self._last_adjust is None:
                self._last_adjust = now
                return
            if now - self._last_adjust < self.interval:
                return
            self._last_adjust = now
            if (
                len(self.analysis_cost_ms) < self.min_samples
                or len(self.capture_cost_ms) < self.min_samples
            ):
                self.reason = "amostras_insuficientes"
                return

            p90_analysis = percentile(self.analysis_cost_ms, 0.90)
            p90_capture = percentile(self.capture_cost_ms, 0.90)
            p90_age = percentile(self.frame_age_ms, 0.90)

            # 20% de margem para gravação, renderização, scheduler e picos.
            analysis_capacity = stepped_limit(
                min(self.max_fps, 1000.0 / max(1, 1.20 * p90_analysis))
            )
            capture_capacity = stepped_limit(
                min(self.max_fps, 1000.0 / max(1, 1.25 * p90_capture))
            )
            target = min(analysis_capacity, capture_capacity)
            # Quando o resultado chega com atraso grande, evitar aumentar a
            # carga. Não reduzir o FPS por idade elevada isolada da latência.
            age_congested = p90_age > max(180.0, 3 * p90_analysis)

            old = (self.capture_fps, self.analysis_fps)
            if target < self.analysis_fps:
                self.analysis_fps = next_step(self.analysis_fps, -1)
                self._good_windows = 0
                self.reason = "reduzindo_carga"
            elif target > self.analysis_fps and not age_congested:
                self._good_windows += 1
                self.reason = "margem_disponivel"
                if self._good_windows >= 2:
                    self.analysis_fps = next_step(self.analysis_fps, +1)
                    self._good_windows = 0
            else:
                self._good_windows = 0
                self.reason = (
                    "fila_atrasada" if age_congested else "capacidade_estavel"
                )

            self.analysis_fps = min(
                self.analysis_fps, self.max_fps, capture_capacity
            )
            # Capturar 30/60 FPS e descartar quase todos os quadros é desperdício.
            # Uma margem de ~25% ajuda a sempre usar o quadro mais atual.
            desired_capture = stepped_limit(
                min(
                    self.max_fps,
                    capture_capacity,
                    max(
                        self.analysis_fps,
                        math.ceil(self.analysis_fps * 1.25),
                    ),
                )
            )
            self.capture_fps = max(self.analysis_fps, desired_capture)
            if old != (self.capture_fps, self.analysis_fps):
                self.adjustments += 1

    def rates(self):
        with self._lock:
            return self.capture_fps, self.analysis_fps

    def snapshot(self):
        with self._lock:
            return {
                "enabled": True,
                "method": "latencia_efetiva_cpu_gpu",
                "gpu_utilization_measured": False,
                "capture_target_fps": self.capture_fps,
                "analysis_target_fps": self.analysis_fps,
                "analysis_p90_ms": round(percentile(self.analysis_cost_ms, 0.90), 2),
                "capture_p90_ms": round(percentile(self.capture_cost_ms, 0.90), 2),
                "age_p90_ms": round(percentile(self.frame_age_ms, 0.90), 2),
                "adjustments": self.adjustments,
                "reason": self.reason,
            }
