"""Teste visual V1 com fonte fictícia. Executar no Linux com xvfb-run.

Não abre BlueStacks e não envia comandos ao jogo. Exercita o botão único,
gravação e encerramento no mesmo fluxo da interface real.
"""

from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import Brawlla  # noqa: E402

raw = np.full((720, 1280, 3), (80, 100, 75), np.uint8)
cv2.putText(
    raw,
    "BRAWLIA V1 SMOKE",
    (360, 240),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (220, 230, 240),
    2,
)


class Source:
    def __init__(self):
        self.source_label = "FONTE SINTETICA - SEM JOGO"
        self.quality_flags = []
        self._last_rect = None

    def grab(self):
        return raw.copy()

    def close(self):
        pass


def advance(root, duration):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        root.update()
        time.sleep(0.02)


with tempfile.TemporaryDirectory() as tmp:
    with (
        patch("brawlla.session.user_data_directory", return_value=Path(tmp)),
        patch("app.AutoWindowCapture", Source),
    ):
        app = Brawlla()
        try:
            # Só a ação de captura é exposta como botão interativo.
            assert app.start_btn.winfo_exists()
            app._toggle()
            advance(app, 1.6)
            assert app.active
            assert app.session.pipeline.adaptive is not None
            assert app.last_data is not None
            app._toggle()
            until = time.monotonic() + 15
            while app.session.state != "idle" and time.monotonic() < until:
                advance(app, 0.1)
            assert app.session.state == "idle"
            assert app.session.last_summary["history"]["complete"]
            assert app.session.last_summary["pipeline"]["adaptive_fps"]["enabled"]
            print("V1 smoke: captura, FPS adaptativo e finalização aprovados")
        finally:
            if app.session.active:
                app.session.stop()
            app.session.join(5)
            try:
                app.destroy()
            except Exception:
                pass
