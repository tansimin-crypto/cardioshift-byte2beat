"""Download and extract the immutable official UCI Heart Disease archive."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.data import CENTER_SOURCES

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARCHIVE = RAW_DIR / "heart+disease.zip"
CHECKSUMS = ROOT / "data" / "checksums.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        request = urllib.request.Request(
            URL,
            headers={"User-Agent": "CardioShift research downloader/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            ARCHIVE.write_bytes(response.read())

    with zipfile.ZipFile(ARCHIVE) as bundle:
        members = {Path(name).name: name for name in bundle.namelist()}
        for source in CENTER_SOURCES:
            if source.filename not in members:
                raise KeyError(f"{source.filename} not found in UCI archive")
            destination = RAW_DIR / source.filename
            with bundle.open(members[source.filename]) as source_handle:
                destination.write_bytes(source_handle.read())

    files = [ARCHIVE, *(RAW_DIR / item.filename for item in CENTER_SOURCES)]
    manifest = {
        "source_url": URL,
        "dataset_doi": "10.24432/C52P4X",
        "license": "CC BY 4.0",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    CHECKSUMS.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    download()
