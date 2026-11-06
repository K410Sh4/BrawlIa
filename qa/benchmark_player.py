"""Compara a recuperação luminosa no mesmo vídeo, sem enviar comandos ao jogo.

Uso: python qa/benchmark_player.py gravacao.mp4 --every-n 10 --limit 150
O vídeo anotado/compactado produz métricas de candidatos, não acurácia.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brawlla.config import VisionConfig  # noqa: E402
from brawlla.health import FrameEvidence  # noqa: E402
from brawlla.player import PlayerLocator  # noqa: E402


def benchmark(video, every_n=10, limit=150):
    if every_n < 1 or limit < 1:
        raise ValueError("every_n e limit devem ser positivos")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Vídeo inválido: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    locators = {
        "regular": PlayerLocator(
            replace(VisionConfig(), player_highlight_fallback=False)
        ),
        "highlight": PlayerLocator(VisionConfig()),
    }
    metrics = {
        key: {"candidate_count": 0, "highlight_used": 0, "times_ms": []}
        for key in locators
    }
    frames_seen = 0
    selected = 0
    try:
        while selected < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id = frames_seen
            frames_seen += 1
            if frame_id % every_n:
                continue
            selected += 1
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            evidence = FrameEvidence(hsv, VisionConfig())
            timestamp = frame_id / fps if fps > 0 else frame_id / 15
            for name, locator in locators.items():
                start = time.perf_counter()
                candidate = locator.locate(hsv, timestamp, evidence=evidence)
                metrics[name]["times_ms"].append(
                    (time.perf_counter() - start) * 1000
                )
                metrics[name]["candidate_count"] += candidate is not None
                metrics[name]["highlight_used"] += bool(
                    locator.diagnostics.get("highlight_fallback_used")
                )
    finally:
        cap.release()
    for item in metrics.values():
        latency = item.pop("times_ms")
        item["mean_ms"] = round(statistics.mean(latency), 3) if latency else None
        item["p95_ms"] = (
            round(sorted(latency)[min(len(latency) - 1, int(len(latency) * 0.95))], 3)
            if latency
            else None
        )
    return {
        "source": str(video),
        "selected_frames": selected,
        "fps_source": fps,
        "accuracy_measured": False,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--every-n", type=int, default=10)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("benchmark_player.json"))
    args = parser.parse_args()
    result = benchmark(args.video, args.every_n, args.limit)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
