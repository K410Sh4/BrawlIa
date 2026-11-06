"""Teste manual assistido da interface com fonte artificial; rodar sob Xvfb."""

import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import ImageGrab

from app import Brawlla
from brawlla.dataset import load_entries
from brawlla.review_ui import CalibrationWindow, ReviewWindow
from brawlla.state import StateDetector

out = Path(__file__).resolve().parents[1] / ".qa_output"
out.mkdir(parents=True, exist_ok=True)
raw = np.full((720, 1280, 3), (80, 110, 50), np.uint8)
cv2.rectangle(raw, (50, 50), (1230, 670), (160, 90, 55), 4)
cv2.putText(
    raw,
    "FONTE ARTIFICIAL - QA",
    (340, 180),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (240, 240, 240),
    2,
)
hud = StateDetector().hud
hud = cv2.resize(
    hud,
    (round(hud.shape[1] * 1180 / 1280), round(hud.shape[0] * 1180 / 1280)),
    interpolation=cv2.INTER_AREA,
)
raw[70 : 70 + hud.shape[0], 240 : 240 + hud.shape[1]] = hud
name = "MEU PLAYER"
tw = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
cv2.putText(
    raw, name, (640 - tw // 2, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 230, 50), 2
)
cv2.rectangle(raw, (600, 286), (680, 300), (95, 45, 45), -1)
cv2.rectangle(raw, (603, 289), (609, 296), (0, 0, 245), -1)
cv2.putText(raw, "1900", (621, 283), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1)
errors = []


class Source:
    def grab(self):
        return raw.copy()

    def close(self):
        pass


def tick(app, duration=0.2):
    until = time.monotonic() + duration
    while time.monotonic() < until:
        app.update()
        time.sleep(0.01)


with (
    tempfile.TemporaryDirectory() as tmp,
    patch("brawlla.session.user_data_directory", return_value=Path(tmp)),
):
    app = Brawlla()
    try:
        app.report_callback_exception = lambda cls, value, tb: errors.append(
            repr(value)
        )
        app._source_factory = lambda: Source
        tick(app)
        calibration = CalibrationWindow(app, raw, app._apply_roi)
        tick(app)
        canvas = calibration.canvas

        # Gestos em coordenadas de tela convertidos para os pixels da fonte.
        def point(x, y):
            return SimpleNamespace(
                x=x * canvas.scale + canvas.offset[0],
                y=y * canvas.scale + canvas.offset[1],
            )

        canvas.begin(point(50, 40))
        canvas.drag(point(1230, 680))
        canvas.end(point(1230, 680))
        calibration.apply()
        assert abs(app.roi[0] - 50 / 1280) < 0.001
        app.record_samples.set(True)
        app.sample_fps.set("2")
        app._start()
        tick(app, 0.9)
        assert app.active
        assert app.last_data["player_health_bar_candidate"] is not None
        assert (
            app.last_data["player_health_bar_candidate"]["health_status"]
            == "low_fill_candidate"
        )
        assert str(app.calibrate_btn.cget("state")) == "disabled"
        app._stop()
        until = time.monotonic() + 8
        while app._finalizing and time.monotonic() < until:
            tick(app, 0.05)
        tick(app, 0.15)
        assert not app._finalizing
        assert app.session.last_summary["video"]["verified"]
        samplefolder = Path(app.session.last_summary["samples"]["folder"])
        assert len(load_entries(samplefolder)) > 0
        review = ReviewWindow(app, samplefolder)
        tick(app)
        review.category.set("jogador")
        review.add_box([100, 100, 50, 60])
        review.status.set("Corrigida")
        assert review.save()
        saved = json.loads(next((samplefolder / "labels").glob("*.json")).read_text())
        assert saved["boxes"][-1]["bbox"] == [100, 100, 50, 60]
        assert saved["reviewed_by"] == "human"
        body_id = saved["boxes"][-1]["entity_id"]
        review.owner_choice.set(
            next(key for key, value in review.owner_map.items() if value == body_id)
        )
        review.category.set("barra_vida")
        review.add_box([100, 80, 50, 5])
        assert review.save()
        saved = json.loads(next((samplefolder / "labels").glob("*.json")).read_text())
        assert saved["boxes"][-1]["owner_id"] == body_id
        assert app.session.last_summary["history"]["complete"]
        assert (
            app.session.last_summary["history"]["records"]
            == app.session.last_summary["pipeline"]["analyzed"]
        )
        tick(app)
        ImageGrab.grab(xdisplay=os.environ["DISPLAY"]).save(out / "ui_v09_revisao.png")
        review.close()
        tick(app)
        ImageGrab.grab(xdisplay=os.environ["DISPLAY"]).save(
            out / "ui_v09_principal.png"
        )
        assert not errors, errors
        app.destroy()
        result = {
            "tk_callback_errors": errors,
            "calibration": "passed",
            "recorded_png": len(load_entries(samplefolder)),
            "video_verified": True,
            "review_roundtrip": "passed",
            "bar_owner_relation": "passed",
            "full_history": "passed",
        }
        (out / "ui_v09.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
    finally:
        if app.active:
            app.session.stop()
        app.session.join(5)
        try:
            app.destroy()
        except Exception:
            pass
