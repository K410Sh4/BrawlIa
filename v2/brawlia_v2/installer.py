"""Small independent updater executable; replaces the GUI after it exits."""

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time

from .updates import ASSET_NAME, sha256_file


def wait_for_exit(pid, seconds=45):
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return
    try:
        if kernel32.WaitForSingleObject(handle, int(seconds * 1000)) != WAIT_OBJECT_0:
            raise TimeoutError("O BrawlIA não fechou para a atualização")
    finally:
        kernel32.CloseHandle(handle)


def apply_update(target, staged, sha256, *, pid=None, relaunch=True):
    target = Path(target).resolve()
    staged = Path(staged).resolve()
    if target.name.lower() != ASSET_NAME.lower():
        raise ValueError("Arquivo de destino não é o executável esperado")
    if not staged.is_file() or staged.parent.parent != target.parent / ".updates":
        raise ValueError("Atualização fora da pasta autorizada")
    if sha256_file(staged) != sha256:
        raise ValueError("O arquivo baixado não passou na verificação SHA-256")
    if pid is not None:
        wait_for_exit(pid)
    replacement = target.with_name("BrawlIA-V2.new.exe")
    backup = target.with_name("BrawlIA-V2.previous.exe")
    import shutil
    shutil.copy2(staged, replacement)
    if sha256_file(replacement) != sha256:
        replacement.unlink(missing_ok=True)
        raise ValueError("Arquivo copiado não passou na verificação")
    # Windows AV and shell can hold a handle briefly after GUI exit.
    for attempt in range(20):
        try:
            if backup.exists():
                backup.unlink()
            target.replace(backup)
            try:
                replacement.replace(target)
            except OSError:
                backup.replace(target)
                raise
            break
        except OSError:
            if attempt == 19:
                raise
            time.sleep(0.4)
    if relaunch:
        subprocess.Popen([str(target)], cwd=str(target.parent), close_fds=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description="BrawlIA V2: atualizador independente")
    parser.add_argument("--apply", action="store_true", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("Atualização executável disponível somente no Windows")
    try:
        apply_update(args.target, args.staged, args.sha256, pid=args.pid)
    except Exception as exc:
        error_file = args.target.parent / "falha_atualizacao.txt"
        error_file.write_text(str(exc), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
