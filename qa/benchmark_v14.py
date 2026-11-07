"""Benchmark A/B de percepção offline usando o mesmo vídeo e as mesmas frames.

Uso: python qa/benchmark_v14.py analise.mp4 --limit 150
Não envia comandos ao jogo. FPS do vídeo não é precisão do detector.
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
from brawlla.player import PlayerLocator  # noqa: E402
from brawlla.terrain import TerrainObserver  # noqa: E402
from brawlla.vision import detect  # noqa: E402


def quantile(samples, q):
    if not samples:
        return None
    return round(sorted(samples)[min(len(samples) - 1, int(len(samples) * q))], 3)


def evaluate(video, limit=150):
    if limit < 1:
        raise ValueError("limit deve ser positivo")
    results = {}
    for label, interval in (("terreno_em_todos", 1), ("terreno_temporal", 3)):
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"Arquivo de vídeo inválido: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        config = replace(VisionConfig(), terrain_sample_every=interval)
        terrain = TerrainObserver(config)
        player = PlayerLocator(config)
        timings, terrain_ms = [], []
        detected = 0
        reviewed = 0
        elapsed = 0.0
        try:
            while reviewed < limit:
                ok, frame = capture.read()
                if not ok:
                    break
                started = time.perf_counter()
                observation, _ = detect(
                    frame, player, elapsed, config, terrain_observer=terrain
                )
                timings.append((time.perf_counter() - started) * 1000)
                terrain_ms.append(observation["vision_stage_timings_ms"]["vision_terrain"])
                detected += observation["player_health_bar_candidate"] is not None
                reviewed += 1
                elapsed += 1.0 / fps
        finally:
            capture.release()
        results[label] = {
            "frames": reviewed,
            "player_candidate_frames": detected,
            "mean_ms": round(statistics.mean(timings), 3) if timings else None,
            "p95_ms": quantile(timings, 0.95),
            "terrain_mean_ms": (
                round(statistics.mean(terrain_ms), 3) if terrain_ms else None
            ),
            "terrain_refreshes": terrain.refreshes,
            "terrain_reuses": terrain.reuses,
        }
    results["methodology"] = {
        "same_video": str(video),
        "terrain_approximate_between_refreshes": True,
        "ground_truth_accuracy_measured": False,
        "capture_gpu_latency_measured": False,
        "note": "O MP4 pode repetir quadros; comparar custo não demonstra acurácia.",
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("benchmark_v14.json"))
    args = parser.parse_args()
    result = evaluate(args.video, args.limit)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
