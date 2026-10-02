"""Small HTTPS helpers.

Uses the operating system trust store (via ``truststore``) so downloads work
behind TLS-inspecting antivirus or corporate proxies that install their own root
certificate.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

_TRUSTSTORE_READY = False
USER_AGENT = "text2sql-bench/0.1"


def _ensure_truststore() -> None:
    global _TRUSTSTORE_READY
    if not _TRUSTSTORE_READY:
        import truststore

        truststore.inject_into_ssl()
        _TRUSTSTORE_READY = True


def _open(url: str, timeout: float):
    _ensure_truststore()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - fixed https URLs


def fetch_bytes(url: str, timeout: float = 120.0) -> bytes:
    with _open(url, timeout) as resp:
        return resp.read()


def download_file(url: str, dest: Path, timeout: float = 120.0) -> Path:
    """Stream ``url`` to ``dest``, writing to a temp file first so a failed
    download never leaves a truncated file behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with _open(url, timeout) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    os.replace(tmp, dest)
    return dest


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
