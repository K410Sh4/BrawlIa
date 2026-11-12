"""Auditoria somente-leitura dos arquivos produzidos por DatasetStore.

Não modifica, repara ou exclui arquivos de sessões.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def reconcile_dataset_integrity(
    folder: Path, *, verify_hashes: bool = True
) -> dict:
    """Confere manifest.jsonl e imagens sem confiar em caminhos do manifesto."""
    root = Path(folder).resolve()
    image_root = root / "images"
    report = {
        "manifest_entries": 0,
        "images_found": 0,
        "missing_images": [],
        "orphan_images": [],
        "hash_mismatches": [],
        "invalid_entries": [],
        "duplicate_paths": [],
    }
    referenced: set[str] = set()
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        report["invalid_entries"].append("manifest.jsonl ausente")
    else:
        with manifest.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        raise ValueError("entrada nao e objeto")
                    name = entry["image"]
                    if not isinstance(name, str):
                        raise ValueError("image nao e string")
                    relative = Path(name)
                    # Rejeitar caminhos absolutos, traversal e arquivos fora de images.
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or len(relative.parts) != 2
                        or relative.parts[0] != "images"
                        or relative.suffix.lower() != ".png"
                    ):
                        raise ValueError("caminho de imagem invalido")
                    path = (root / relative).resolve()
                    if not path.is_relative_to(image_root.resolve()):
                        raise ValueError("imagem fora do diretorio permitido")
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    report["invalid_entries"].append(f"linha {number}: {exc}")
                    continue
                report["manifest_entries"] += 1
                normalized = relative.as_posix()
                if normalized in referenced:
                    report["duplicate_paths"].append(normalized)
                referenced.add(normalized)
                if not path.is_file():
                    report["missing_images"].append(normalized)
                    continue
                if verify_hashes:
                    expected = entry.get("sha256")
                    if not isinstance(expected, str) or len(expected) != 64:
                        report["invalid_entries"].append(
                            f"linha {number}: sha256 invalido"
                        )
                    else:
                        with path.open("rb") as image_stream:
                            digest = hashlib.file_digest(
                                image_stream, "sha256"
                            ).hexdigest()
                        if digest != expected.lower():
                            report["hash_mismatches"].append(normalized)
    if image_root.is_dir():
        for image in sorted(image_root.rglob("*.png")):
            report["images_found"] += 1
            name = image.relative_to(root).as_posix()
            if name not in referenced:
                report["orphan_images"].append(name)
    report["ok"] = not any(
        report[key]
        for key in (
            "missing_images",
            "orphan_images",
            "hash_mismatches",
            "invalid_entries",
            "duplicate_paths",
        )
    )
    return report
