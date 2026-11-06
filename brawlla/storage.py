"""Pastas estáveis, preferências e escrita atômica de diagnósticos locais."""

from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid
from .config import SessionOptions


def application_directory():
    return (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[1]
    )


def user_data_directory():
    if os.name == "nt":
        base = Path(
            os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        )
    else:
        base = Path(
            os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        )
    return (base / "Brawlla").resolve()


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def probe_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    test = path / ("write_check_" + uuid.uuid4().hex + ".tmp")
    try:
        with test.open("xb") as file:
            file.write(b"Brawlla write check")
    finally:
        test.unlink(missing_ok=True)
    return path


def load_preferences(root):
    path = Path(root) / "preferencias.json"
    if not path.exists():
        return SessionOptions(record_video=True)
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(SessionOptions)}:
        raise ValueError("Preferências de sessão inválidas")
    defaults = asdict(SessionOptions(record_video=True))
    return SessionOptions(**{**defaults, **raw})


def save_preferences(root, options):
    if not isinstance(options, SessionOptions):
        raise ValueError("SessionOptions esperado")
    write_json(Path(root) / "preferencias.json", asdict(options))


def new_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]


def utc_now():
    return datetime.now(timezone.utc).isoformat()
