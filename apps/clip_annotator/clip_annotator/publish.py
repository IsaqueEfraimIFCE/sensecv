"""Publish full SenseCV recordings to Drive plus a links-only index.

    python -m clip_annotator.publish --dataset <dir> --index index.sqlite

A full clip is a recording folder as the phone app writes it: one ``.mp4``,
``frames.json`` and ``metadata.json`` (exported cuts carry ``cut_info.json``
and are skipped). Clip id = ``<collection>__<recording>``, e.g.
``SenseCV-10-09-2026-UNIFOR-ISAQUE__41``. The original MP4 is uploaded as-is
(720x480, a few MB each), so no transcoding is needed.

Resumable: clips already in the index are skipped, and uploads are
find-or-replace by name, so a crashed run can simply be started again.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .events import is_safe_id
from .index_db import Clip, IndexDb
from .mp4 import mp4_duration

log = logging.getLogger(__name__)

INDEX_NAME = "index.sqlite"
VIDEO_MIME = "video/mp4"
# Synthetic stress-test recordings are not real walks (same rule as the viewer).
EXCLUDED_COLLECTIONS = {"SenseCV-Teste-Estresse"}
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ClipSource:
    clip_id: str
    location: str
    recording: str
    video: Path
    duration_s: float
    fps: float
    frame_count: int


@dataclass
class Summary:
    published: int = 0
    skipped: int = 0
    failed: list = field(default_factory=list)  # [(clip_id, message)]


def _video_of(folder: Path):
    preferred = folder / "video.mp4"
    if preferred.is_file():
        return preferred
    mp4s = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".mp4" and p.is_file())
    return mp4s[0] if mp4s else None


def _is_full_clip(folder: Path) -> bool:
    return ((folder / "frames.json").is_file() and (folder / "metadata.json").is_file()
            and not (folder / "cut_info.json").exists() and _video_of(folder) is not None)


def read_clip(folder: Path, collection: str) -> ClipSource:
    """Timing of one recording: fps as selected on the phone, frame count from
    ``frames.json`` and duration from the MP4 header - the timeline the browser
    plays. ``metadata.json`` overstates it by ~0.7 s, so it is the last resort
    after the ``frames.json`` span."""
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    frames = json.loads((folder / "frames.json").read_text(encoding="utf-8"))["frames"]
    if not frames:
        raise ValueError(f"{folder}: frames.json lists no frames")
    times = [f["time_usec"] for f in frames]
    span = (times[-1] - times[0]) / 1e6
    fps = float(meta.get("fps_selected") or 0) or ((len(times) - 1) / span if span > 0 else 0.0)
    video = _video_of(folder)
    duration = (mp4_duration(video) or (span + 1.0 / fps if span > 0 and fps else 0.0)
                or float(meta.get("duration_seconds") or 0))
    if fps <= 0 or duration <= 0:
        raise ValueError(f"{folder}: cannot tell fps/duration")
    recording = folder.name
    clip_id = _UNSAFE.sub("_", f"{collection}__{recording}").strip("._")
    if not is_safe_id(clip_id):
        raise ValueError(f"{folder}: clip id {clip_id!r} is not usable")
    return ClipSource(clip_id, collection, recording, video, round(duration, 3),
                      round(fps, 3), len(frames))


def discover_clips(root) -> list:
    """Every full clip under ``root``; the collection is the first folder level."""
    root = Path(root)
    found = []
    for current, dirs, _files in os.walk(root):
        folder = Path(current)
        dirs.sort()
        if not _is_full_clip(folder):
            continue
        dirs[:] = []
        rel = folder.relative_to(root).parts
        collection = rel[0] if len(rel) > 1 else root.name
        if collection in EXCLUDED_COLLECTIONS:
            continue
        try:
            found.append(read_clip(folder, collection))
        except (ValueError, KeyError, TypeError, OSError) as error:
            log.warning("skipping %s: %s", folder, error)
    ids = [c.clip_id for c in found]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate clip ids: {sorted(duplicates)}")
    return sorted(found, key=lambda c: (c.location, c.recording))


def _row(source: ClipSource, video_ref: str) -> Clip:
    return Clip(source.clip_id, source.location, source.recording, source.duration_s,
                source.fps, source.frame_count, video_ref, source.video.stat().st_size)


def build_local_index(root, index: IndexDb) -> int:
    """Development index: ``video_ref`` = path relative to ``root``."""
    index.create()
    known = index.clip_ids()
    added = 0
    for source in discover_clips(root):
        if source.clip_id in known:
            continue
        index.add_clip(_row(source, source.video.relative_to(Path(root)).as_posix()))
        added += 1
    return added


def publish_clip(source: ClipSource, drive, videos_folder_id: str, index: IndexDb) -> None:
    file_id = drive.upload(f"{source.clip_id}.mp4", source.video.read_bytes(), videos_folder_id,
                           VIDEO_MIME)
    index.add_clip(_row(source, file_id))


def publish_all(sources, drive, videos_folder_id: str, index: IndexDb, *, workers: int = 4,
                progress=None) -> Summary:
    summary = Summary()
    known = index.clip_ids()
    pending = [s for s in sources if s.clip_id not in known]
    summary.skipped = len(sources) - len(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(publish_clip, s, drive, videos_folder_id, index): s
                   for s in pending}
        for future in as_completed(futures):
            source = futures[future]
            try:
                future.result()
                summary.published += 1
            except Exception as error:  # noqa: BLE001 - one bad clip must not stop the run
                log.exception("clip %s failed", source.clip_id)
                summary.failed.append((source.clip_id, f"{type(error).__name__}: {error}"))
            if progress:
                progress(summary, len(pending))
    return summary


def upload_index(drive, index_path, folder_id: str) -> str:
    return drive.upload(INDEX_NAME, Path(index_path).read_bytes(), folder_id,
                        "application/vnd.sqlite3")


def main(argv=None) -> int:
    from .drive import DriveClient, credentials_from_env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--index", type=Path, default=Path("index.sqlite"))
    parser.add_argument("--root-name", default="SenseCV clip annotator")
    parser.add_argument("--limit", type=int, help="publish only the first N clips")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-upload-index", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    sources = discover_clips(args.dataset)
    if args.limit:
        sources = sources[: args.limit]
    drive = DriveClient(credentials_from_env(os.environ))
    root = drive.ensure_folder(args.root_name, "root")
    videos_folder = drive.ensure_folder("videos", root)
    annotations_folder = drive.ensure_folder("annotations", root)
    index = IndexDb(args.index)
    index.create()

    def show(summary, total):
        done = summary.published + len(summary.failed)
        if done % 10 == 0 or done == total:
            print(f"{done}/{total} clips ({len(summary.failed)} failed)", file=sys.stderr, flush=True)

    summary = publish_all(sources, drive, videos_folder, index, workers=args.workers, progress=show)
    if not args.no_upload_index:
        upload_index(drive, args.index, videos_folder)
    print(json.dumps({
        "VIDEOS_DRIVE_FOLDER_ID": videos_folder,
        "ANNOTATIONS_DRIVE_FOLDER_ID": annotations_folder,
        "published": summary.published, "skipped": summary.skipped, "failed": summary.failed,
    }, indent=2))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
