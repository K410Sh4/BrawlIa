"""Reprocessa os quadros registrados de um vídeo ANOTADO; não mede acurácia."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("project", type=Path)
parser.add_argument("video", type=Path)
parser.add_argument("log", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--video-map", type=Path, help="video_frames.jsonl da versão 0.10+")
args = parser.parse_args()
sys.path.insert(0, str(args.project.resolve()))
from brawlla.engine import SessionAnalyzer

rows = [
    json.loads(line)
    for line in args.log.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]
mapping_path = args.video_map or args.log.parent / "video_frames.jsonl"
if any(row.get("video_frame") is not None for row in rows):
    lookup = {
        row["video_frame"]: row for row in rows if row.get("video_frame") is not None
    }
else:
    if not mapping_path.exists():
        raise ValueError(
            "Envie video_frames.jsonl junto do histórico ou use --video-map para correspondência exata."
        )
    mapping = [
        json.loads(line)
        for line in mapping_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    by_frame = {row["frame_id"]: row for row in rows}
    lookup = {}
    for mapped in mapping:
        row = by_frame.get(mapped["frame_id"])
        if row is not None and mapped.get("video_frame") is not None:
            if mapped.get("session_id") != row.get("session_id"):
                raise ValueError("Mapa de vídeo pertence a outra sessão")
            lookup[mapped["video_frame"]] = {
                **row,
                **{
                    key: mapped[key]
                    for key in ("video_frame", "video_content_xywh", "input_resolution")
                },
            }
reader = cv2.VideoCapture(str(args.video))
engine = SessionAnalyzer()
results = []
index = 0
try:
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        row = lookup.get(index)
        index += 1
        if row is None:
            continue
        x, y, w, h = row["video_content_xywh"]
        image = cv2.resize(frame[y : y + h, x : x + w], tuple(row["input_resolution"]))
        start = time.perf_counter()
        _, d = engine.analyze(image, row["elapsed_s"], render=False)
        elapsed = (time.perf_counter() - start) * 1000
        p = d.get("player_health_bar_candidate")
        results.append(
            {
                "time": row["elapsed_s"],
                "player": p,
                "state": d["game_state"]["name"],
                "player_ids": [
                    o["track_id"]
                    for o in d["tracked_objects"]
                    if o["kind"] == "jogador_candidato"
                ],
                "camera": d.get("camera_motion"),
                "objects": len(d["tracked_objects"]),
                "elapsed_ms": round(elapsed, 3),
                "diagnostics": d.get("player_detection_diagnostics"),
            }
        )
        if args.limit and len(results) >= args.limit:
            break
finally:
    reader.release()
combat = [r for r in results if r["state"] == "combate"]
present = [r for r in combat if r["player"]]
summary = {
    "source_is_annotated": True,
    "frames": len(results),
    "combat": len(combat),
    "player_present": len(present),
    "coverage_not_accuracy": (
        round(100 * len(present) / len(combat), 2) if combat else None
    ),
    "player_ids": len({tid for r in results for tid in r["player_ids"]}),
    "mean_ms": round(sum(r["elapsed_ms"] for r in results) / max(1, len(results)), 3),
    "camera_valid": sum(bool((r["camera"] or {}).get("valid")) for r in results),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps({"summary": summary, "frames": results}, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False))
