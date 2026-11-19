"""Пакет артефактов модели: файлы + manifest.json с хэшами, чтобы потребитель мог проверить целостность."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(bundle: Path, meta: dict) -> Path:
    files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(bundle.iterdir()) if p.is_file() and p.name != "manifest.json"}
    manifest = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), **meta, "files": files}
    out = bundle / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return out


def verify(bundle: Path) -> dict:
    """Проверяет хэши всех файлов из манифеста. Бросает ValueError при расхождении."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        if sha256(bundle / name) != info["sha256"]:
            raise ValueError(f"повреждён или подменён файл артефакта: {name}")
    return manifest
