"""Configuração efetiva e versões necessárias para reproduzir uma sessão."""

from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import sys

from .config import VERSION


def provenance_snapshot(config):
    encoded = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    packages = {}
    for name in ("opencv-python", "opencv-python-headless", "numpy", "Pillow", "mss"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {
        "app_version": VERSION,
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "packages": packages,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
    }
