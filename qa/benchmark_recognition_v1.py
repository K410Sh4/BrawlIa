"""Diagnóstico de desempenho do dataset local Recognition V1 (não mede acurácia)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def percentile95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, (95 * len(ordered) + 99) // 100 - 1))]


def stage_stats(entries, key):
    values = [float(e[key]) for e in entries if key in e]
    return {
        "samples": len(values),
        "mean_ms": round(statistics.mean(values), 3) if values else None,
        "p95_ms": round(percentile95(values), 3) if values else None,
    }


def benchmark(folder: Path):
    folder = folder.resolve()
    entries = [
        json.loads(line)
        for line in (folder / "observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError("Nenhum ciclo de reconhecimento armazenado")
    duration = max(0.0, entries[-1]["timestamp_s"] - entries[0]["timestamp_s"])
    time_ms = [float(entry.get("processing_ms", 0)) for entry in entries]
    acquired = sum(int(e.get("acquired_pixels", 0)) for e in entries)
    possible = sum(int(e.get("full_screen_pixels", 0)) for e in entries)
    modes = Counter(e.get("capture_mode", "unknown") for e in entries)
    events = Counter(
        item.get("event", "unknown") for e in entries for item in e.get("events", [])
    )
    reasons = Counter(e.get("capture_reason", "unknown") for e in entries)
    suppressed = sum(int(e.get("roi_suppressed_targets", 0)) for e in entries)
    hud_scans = sum(bool(e.get("hud_check_performed", False)) for e in entries)
    result = {
        "cycles": len(entries),
        "duration_s": round(duration, 3),
        "effective_fps": round((len(entries) - 1) / duration, 3) if duration else 0,
        "capture_and_analysis_avg_ms": round(statistics.mean(time_ms), 3),
        "capture_and_analysis_p95_ms": round(percentile95(time_ms), 3),
        "full_cycles": modes.get("full", 0),
        "regional_cycles": modes.get("regional", 0),
        "pixels_acquired": acquired,
        "pixels_if_full": possible,
        "pixel_reduction_fraction": (
            round(1 - acquired / possible, 4) if possible else None
        ),
        "events": dict(events),
        "capture_reasons": dict(reasons),
        "suppressed_roi_targets": suppressed,
        "hud_checks": hud_scans,
        "stages": {
            key: stage_stats(entries, key)
            for key in (
                "capture_ms",
                "analysis_ms",
                "hud_check_ms",
                "dataset_enqueue_ms_previous_cycle",
                "processing_ms",
            )
        },
        "dataset_queue_max": max(
            (int(e.get("dataset_queue_depth", 0)) for e in entries),
            default=0,
        ),
        "warning": (
            "Economia de pixels não é economia de CPU nem acurácia. "
            "O tempo inclui aquisição e análise, diferentemente da V1.4."
        ),
    }
    (folder / "benchmark_recognition_v1.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.folder), indent=2, ensure_ascii=False))
