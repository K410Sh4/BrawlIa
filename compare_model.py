"""Executa inferência ONNX independente sobre MP4, sem controlar o BlueStacks."""

import argparse
import json
from pathlib import Path
import statistics

import cv2

from brawlla.model_vision import ModelSchedule, OnnxVision


def run(args):
    classes = ()
    if args.classes:
        data = json.loads(Path(args.classes).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            ids = sorted(int(k) for k in data)
            classes = tuple(str(data.get(str(k), f"class_{k}")) for k in range(max(ids) + 1))
        elif isinstance(data, list):
            classes = tuple(map(str, data))
        else:
            raise ValueError("Classes devem ser lista ou dicionário")
    model = OnnxVision(
        args.model,
        classes=classes,
        provider=args.provider,
        input_size=args.input_size,
        export_format=args.format,
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir {args.video}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    schedule = ModelSchedule(args.every_n)
    count = 0
    observed = 0
    timings = []
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    try:
        with Path(args.output).open("w", encoding="utf-8") as output:
            while True:
                read, frame = capture.read()
                if not read:
                    break
                index = count
                count += 1
                if not schedule.should_infer(index):
                    continue
                prediction = model.detect(frame, confidence=args.confidence, iou=args.iou)
                observed += 1
                timings.append(prediction["inference_ms"])
                # Registro separado para cada quadro observado; sem copiar quadros prévios.
                item = {
                    "source_frame": index,
                    "timestamp_s": round(index / fps, 5) if fps > 0 else None,
                    "resolution": [frame.shape[1], frame.shape[0]],
                    **prediction,
                }
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
    finally:
        capture.release()
    result = {
        "model": str(args.model),
        "provider": model.provider,
        "source_frames": count,
        "inferred_frames": observed,
        "unobserved_frames": count - observed,
        "inference_every_n": args.every_n,
        "mean_inference_ms": round(statistics.mean(timings), 3) if timings else None,
        "p95_inference_ms": (
            round(sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))], 3)
            if timings
            else None
        ),
        "output": str(args.output),
        "accuracy_measured": False,
    }
    summary = Path(args.output).with_suffix(".summary.json")
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="Modelo ONNX fornecido por você")
    parser.add_argument("--classes", help="Lista JSON de classes do modelo")
    parser.add_argument("--output", default="modelo_analise.jsonl")
    parser.add_argument("--provider", choices=["auto", "directml", "cuda", "cpu"], default="auto")
    parser.add_argument("--format", choices=["yolov8", "xyxy"], default="yolov8")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--every-n", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
