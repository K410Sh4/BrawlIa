"""Command-line 30 FPS baseline. Keep the V1 application untouched."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from .capture import MSSBlueStacksSource, SyntheticSource
from .pipeline import Pipeline
from .window import enable_dpi_awareness


def main(argv=None):
    parser = argparse.ArgumentParser(description="BrawlIA V2 / Etapa 1 — captura e telemetria a 30 FPS")
    parser.add_argument("--source", choices=("bluestacks", "synthetic"), default="bluestacks")
    parser.add_argument("--duration", type=float, default=30.0, help="Duração em segundos (0 = até Ctrl+C)")
    parser.add_argument("--output", type=Path, default=Path("saida"), help="Pasta do relatório JSON")
    args = parser.parse_args(argv)
    if args.duration < 0:
        parser.error("--duration não pode ser negativa")
    enable_dpi_awareness()
    source = MSSBlueStacksSource if args.source == "bluestacks" else SyntheticSource
    pipeline = Pipeline(source).start()
    print("BrawlIA V2 / Fase 1 — medição de FPS reais; SEM OCR, vídeo, PNG ou automação")
    print("A janela do BlueStacks deve permanecer visível. Pressione Ctrl+C para encerrar.")
    started = time.monotonic()
    try:
        while not pipeline.stop_event.wait(1.0):
            report = pipeline.snapshot()
            print(
                "Captura {capture_recent_fps:5.1f} | processamento {process_recent_fps:5.1f} FPS "
                "| captura p95 {capture_p95_ms:5.1f} ms | atraso p95 {frame_age_p95_ms:5.1f} ms "
                "| substituídos {replaced_before_processing}".format(**report),
                flush=True,
            )
            if args.duration and time.monotonic() - started >= args.duration:
                break
    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")
    report = pipeline.stop()
    args.output.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("v2_fase1_%Y%m%d_%H%M%S.json")
    destination = args.output / name
    report["source"] = args.source
    report["phase"] = "capture_only"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Métricas salvas: {destination.resolve()}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["error"]:
        return 1
    if not report["capture_count"]:
        print("Nenhum quadro capturado — confirme a janela do emulador.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
