"""Compila o aplicativo no Windows usando a versão central do pacote."""

import os
from pathlib import Path
import subprocess
import sys
from brawlla.config import VERSION_SHORT

if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("Compile o EXE no Windows.")
    os.chdir(Path(__file__).resolve().parent)
    name = f"Brawlla-V{VERSION_SHORT}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--windowed",
            "--name",
            name,
            "--add-data",
            f"brawlla/assets{os.pathsep}brawlla/assets",
            "app.py",
        ],
        check=True,
    )
    print(f"EXE gerado em dist/{name}.exe")
