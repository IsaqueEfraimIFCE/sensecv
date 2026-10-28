"""Import raw SenseCV collections from Drive zips straight onto the server volume.

    python -m clip_annotator.drive_import --zips zips.json --token-file token.json \
        --raw /data/raw --data /data

``zips.json`` lists ``[{"id": "<drive file id>", "name": "SenseCV-...zip"}]``;
the token file holds ``{"access_token": "..."}`` - a ~1 h bearer token, never a
refresh token, so nothing long-lived ever sits on the server.

Per zip: download into ``<raw>/_zips``, extract only what the annotator needs
from each recording (its ``.mp4``, ``frames.json``, ``metadata.json`` and, so
exported cuts stay recognisable, ``cut_info.json``) into
``<raw>/<collection>/...``, delete the zip and mark the collection complete.
Then every recording is indexed with ``build_bundle(link=True)``: videos are
hard-linked into ``<data>/videos``, so they take no extra space. Resumable.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

from .publish import build_bundle

log = logging.getLogger(__name__)

KEEP_NAMES = {"frames.json", "metadata.json", "cut_info.json"}
DONE_MARKER = ".complete"
# A zip needs room for itself plus what is extracted from it (~the same size).
SPACE_FACTOR = 2.2
CHUNK_BYTES = 64 * 1024 * 1024


def _wanted(name: str) -> bool:
    base = PurePosixPath(name).name
    return base in KEEP_NAMES or base.lower().endswith(".mp4")


def _safe_parts(name: str):
    """Path parts of a zip member, or None for anything that is not a plain
    relative path (absolute, ``..``, macOS resource forks)."""
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if not parts or name.startswith("/") or any(p in ("..", "") for p in parts):
        return None
    if parts[0] == "__MACOSX" or parts[-1].startswith("._"):
        return None
    return parts


def extract_recordings(zip_path, dest) -> int:
    """Extract the annotator's files from one collection zip; returns files written."""
    dest = Path(dest)
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not _wanted(info.filename):
                continue
            parts = _safe_parts(info.filename)
            if parts is None:
                log.warning("skipping unsafe or junk member %r", info.filename)
                continue
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK_BYTES)
            written += 1
    return written


def run_import(zips, raw_dir, data_dir, download) -> dict:
    """``download(file_id, path)`` fetches one Drive file to ``path``."""
    raw = Path(raw_dir)
    staging = raw / "_zips"
    staging.mkdir(parents=True, exist_ok=True)
    report = {"imported": [], "skipped": [], "failed": []}
    for entry in zips:
        name = entry["name"]
        collection = name[:-4] if name.lower().endswith(".zip") else name
        folder = raw / collection
        if (folder / DONE_MARKER).exists():
            report["skipped"].append(collection)
            continue
        size = int(entry.get("size", 0))
        free = shutil.disk_usage(raw).free
        if size and free < size * SPACE_FACTOR:
            report["failed"].append((collection, f"needs ~{size * SPACE_FACTOR / 1e9:.1f} GB free, "
                                                 f"has {free / 1e9:.1f} GB"))
            continue
        zip_path = staging / name
        try:
            download(entry["id"], zip_path)
            files = extract_recordings(zip_path, folder)
            (folder / DONE_MARKER).write_text(str(files), encoding="utf-8")
            report["imported"].append((collection, files))
            print(f"imported {collection}: {files} files", flush=True)
        except Exception as error:  # noqa: BLE001 - one bad zip must not stop the run
            log.exception("collection %s failed", collection)
            report["failed"].append((collection, f"{type(error).__name__}: {error}"))
        finally:
            zip_path.unlink(missing_ok=True)
    report["indexed"] = build_bundle(raw, data_dir, link=True)
    return report


def drive_downloader(access_token: str):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    service = build("drive", "v3", credentials=Credentials(token=access_token), cache_discovery=False)

    def download(file_id: str, path: Path) -> None:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with open(path, "wb") as handle:
            downloader = MediaIoBaseDownload(handle, request, chunksize=CHUNK_BYTES)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=5)

    return download


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--raw", type=Path, default=Path("/data/raw"))
    parser.add_argument("--data", type=Path, default=Path("/data"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    token = json.loads(args.token_file.read_text(encoding="utf-8"))["access_token"]
    report = run_import(json.loads(args.zips.read_text(encoding="utf-8")), args.raw, args.data,
                        drive_downloader(token))
    print(json.dumps(report, indent=1), flush=True)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
