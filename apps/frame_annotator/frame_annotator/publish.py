"""Publish the dataset to Drive as per-clip sprite sheets plus a links-only index.

    python -m frame_annotator.publish --dataset <dir> --index index.sqlite

Resumable: clips already in the index are skipped, and sheet uploads are
find-or-replace by name, so a crashed run can simply be started again.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .index_db import Clip, IndexDb, Sheet
from .sprites import DEFAULT_COLS, DEFAULT_QUALITY, DEFAULT_ROWS, DEFAULT_TILE, build_sheets

log = logging.getLogger(__name__)
INDEX_NAME = "index.sqlite"


@dataclass(frozen=True)
class ClipJob:
    clip_id: str
    location: str
    frames: tuple  # Paths, in frame order


@dataclass
class Summary:
    published: int = 0
    skipped: int = 0
    failed: list = field(default_factory=list)  # [(clip_id, message)]


def jobs_from_manifest(dataset_dir, manifest) -> list:
    """One job per clip. Clip id = ``<env>_<MMDD>_<NN>``, taken from the file names."""
    groups = defaultdict(list)
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prefix = row["image"].rsplit("_", 1)[0]
            groups[(row["env"], row["env_name"], prefix)].append((int(row["frame_index"]), row["path"]))
    jobs = []
    for (env, env_name, prefix), frames in sorted(groups.items()):
        frames.sort()
        if [index for index, _ in frames] != list(range(len(frames))):
            raise ValueError(f"clip {env}_{prefix}: frame indices are not contiguous from 0")
        jobs.append(ClipJob(
            f"{env}_{prefix}", f"{env}_{env_name}", tuple(Path(dataset_dir) / p for _, p in frames)
        ))
    return jobs


def publish_clip(job: ClipJob, drive, sheets_folder_id: str, index: IndexDb, **sheet_options) -> int:
    sheets = build_sheets(job.frames, **sheet_options)
    rows = []
    for sheet in sheets:
        name = f"{job.clip_id}__s{sheet.sheet_index:02d}.jpg"
        file_id = drive.upload(name, sheet.jpeg, sheets_folder_id, "image/jpeg")
        rows.append(Sheet(job.clip_id, sheet.sheet_index, file_id, sheet.first_frame,
                          sheet.frame_count, sheet.cols, sheet.tile_w, sheet.tile_h))
    index.add_clip(Clip(job.clip_id, job.location, len(job.frames)), rows)
    return len(rows)


def publish_all(jobs, drive, sheets_folder_id: str, index: IndexDb, *, workers: int = 4,
                progress=None, **sheet_options) -> Summary:
    summary = Summary()
    pending = []
    for job in jobs:
        if index.clip(job.clip_id) is not None:
            summary.skipped += 1
        else:
            pending.append(job)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(publish_clip, job, drive, sheets_folder_id, index, **sheet_options): job
            for job in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                summary.published += 1
            except Exception as error:  # noqa: BLE001 - one bad clip must not stop the run
                log.exception("clip %s failed", job.clip_id)
                summary.failed.append((job.clip_id, f"{type(error).__name__}: {error}"))
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
    parser.add_argument("--manifest", type=Path, help="default: <dataset>/manifest.csv")
    parser.add_argument("--index", type=Path, default=Path("index.sqlite"))
    parser.add_argument("--root-name", default="SenseCV frame annotator")
    parser.add_argument("--limit", type=int, help="publish only the first N clips")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--no-upload-index", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    drive = DriveClient(credentials_from_env(os.environ))
    root = drive.ensure_folder(args.root_name, "root")
    sheets_folder = drive.ensure_folder("sheets", root)
    annotations_folder = drive.ensure_folder("annotations", root)

    index = IndexDb(args.index)
    index.create()
    jobs = jobs_from_manifest(args.dataset, args.manifest or args.dataset / "manifest.csv")
    if args.limit:
        jobs = jobs[: args.limit]

    def show(summary, total):
        done = summary.published + len(summary.failed)
        if done % 25 == 0 or done == total:
            print(f"{done}/{total} clips ({len(summary.failed)} failed)", file=sys.stderr, flush=True)

    summary = publish_all(jobs, drive, sheets_folder, index, workers=args.workers, progress=show,
                          tile=args.tile, cols=args.cols, rows=args.rows, quality=args.quality)
    if not args.no_upload_index:
        upload_index(drive, args.index, sheets_folder)
    print(json.dumps({
        "SHEETS_DRIVE_FOLDER_ID": sheets_folder,
        "ANNOTATIONS_DRIVE_FOLDER_ID": annotations_folder,
        "published": summary.published, "skipped": summary.skipped, "failed": summary.failed,
    }, indent=2))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
