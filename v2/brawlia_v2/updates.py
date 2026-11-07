"""Updates over HTTPS from this project's trusted GitHub Releases.

Only stable releases with the exact version format v2.major.patch are accepted.
The manifest and .exe are both fetched from the same release; the SHA-256
prevents truncated/corrupted downloads. This is not code signing.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import __version__

REPO = "K410Sh4/BrawlIa"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
MANIFEST_NAME = "ota-manifest.json"
ASSET_NAME = "BrawlIA-V2.exe"
MAX_EXECUTABLE_BYTES = 250 * 1024 * 1024
VERSION_RE = re.compile(r"^v?(2)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
ALLOWED_DOWNLOAD_HOSTS = {
    "github.com",
    "api.github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
}


@dataclass(frozen=True)
class Release:
    version: str
    manifest_url: str
    executable_url: str
    html_url: str


def version_tuple(text):
    match = VERSION_RE.fullmatch(str(text))
    if match is None:
        raise ValueError("Versão V2 inválida")
    return tuple(int(value) for value in match.groups())


def validate_https_url(url):
    parts = urlparse(url)
    if (parts.scheme != "https" or parts.hostname not in ALLOWED_DOWNLOAD_HOSTS
            or parts.username or parts.password or not parts.path):
        raise ValueError("Origem de atualização não autorizada")
    return url


def _get(url, max_bytes=65536):
    validate_https_url(url)
    req = Request(url, headers={
        "User-Agent": "BrawlIA-V2-Update/2",
        "Accept": "application/vnd.github+json",
    })
    with urlopen(req, timeout=15) as response:
        validate_https_url(response.geturl())
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Resposta da atualização excedeu o limite")
    return data


def parse_latest(data, current=__version__):
    if not isinstance(data, dict) or data.get("draft") or data.get("prerelease"):
        return None
    tag = data.get("tag_name", "")
    try:
        if version_tuple(tag) <= version_tuple(current):
            return None
    except ValueError:
        return None
    items = data.get("assets", [])
    if not isinstance(items, list):
        return None
    url_by_name = {asset.get("name"): asset.get("browser_download_url")
                   for asset in items if isinstance(asset, dict)}
    if not all(isinstance(url_by_name.get(name), str)
               for name in (MANIFEST_NAME, ASSET_NAME)):
        return None
    manifest_url = validate_https_url(url_by_name[MANIFEST_NAME])
    exe_url = validate_https_url(url_by_name[ASSET_NAME])
    html_url = validate_https_url(data.get("html_url", ""))
    return Release(tag.lstrip("v"), manifest_url, exe_url, html_url)


def check_release(current=__version__):
    info = json.loads(_get(API_URL))
    return parse_latest(info, current)


def validate_manifest(data, release):
    if not isinstance(data, dict):
        raise ValueError("Manifesto de atualização inválido")
    if (data.get("version") != release.version
            or data.get("filename") != ASSET_NAME):
        raise ValueError("Manifesto não corresponde à versão publicada")
    digest = data.get("sha256")
    size = data.get("size")
    if (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or type(size) is not int or not 1 <= size <= MAX_EXECUTABLE_BYTES):
        raise ValueError("Manifesto contém hash ou tamanho inválido")
    return digest, size


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(release, directory):
    """Download to a private staging directory. Never overwrite an app in use."""
    manifest = json.loads(_get(release.manifest_url))
    digest, expected_size = validate_manifest(manifest, release)
    base = Path(directory).resolve()
    base.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".v2-download-", suffix=".tmp", dir=base)
    size = 0
    try:
        with os.fdopen(fd, "wb") as target:
            req = Request(release.executable_url,
                          headers={"User-Agent": "BrawlIA-V2-Update/2"})
            with urlopen(req, timeout=30) as response:
                validate_https_url(response.geturl())
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > expected_size or size > MAX_EXECUTABLE_BYTES:
                        raise ValueError("Atualização ultrapassou o tamanho esperado")
                    target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if size != expected_size or sha256_file(tmp_name) != digest:
            raise ValueError("SHA-256/tamanho da atualização não confere")
        staged = base / f"BrawlIA-V2-{release.version}.exe"
        os.replace(tmp_name, staged)
        return staged, digest
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
