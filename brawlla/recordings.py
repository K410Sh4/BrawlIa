"""Navegação segura para a pasta de gravações do BrawlIA.

Não executa comandos no jogo. Nunca abre caminhos provenientes de relatórios
externos: a seleção é limitada à árvore de gravações do próprio aplicativo.
"""

import os
from pathlib import Path
import subprocess
import sys


def _sessions_root(data_root):
    return (Path(data_root) / "gravacoes").resolve()


def recording_folder(data_root, current=None):
    """Sessão atual, última sessão salva ou diretório geral.

    Compatível com reinicialização do aplicativo: a última sessão local
    permanece localizável mesmo sem estado persistido na interface.
    """
    parent = _sessions_root(data_root)
    if current is not None:
        preferred = Path(current)
        if preferred.is_symlink():
            raise ValueError("Pasta da sessão não pode ser link simbólico")
        preferred = preferred.resolve()
        if preferred.parent != parent:
            raise ValueError("Pasta da sessão está fora de gravacoes")
        if preferred.is_dir():
            return preferred
    if parent.is_dir():
        folders = (
            entry
            for entry in parent.iterdir()
            if entry.is_dir()
            and not entry.is_symlink()
            and (entry / "sessao.json").is_file()
        )
        latest = max(folders, key=lambda folder: folder.name, default=None)
        if latest is not None:
            return latest
    return parent


def open_recording_folder(path):
    """Abre pasta no Explorer/Finder/gerenciador local, sem shell=True."""
    folder = Path(path).resolve(strict=True)
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))
    if sys.platform == "win32":
        os.startfile(str(folder))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])
