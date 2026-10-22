"""Video sources: map a clip's ``video_ref`` to a local file the server can stream.

The browser seeks an MP4 with HTTP Range requests, so the app serves files
(``send_file`` answers ranges with 206); it never holds whole videos in memory.

* ``CachedVideoSource`` - production: Drive file id -> size-capped disk cache.
  Browsers fire several range requests at once, so each file is downloaded once
  under its own lock.
* ``LocalVideoSource`` - development: path relative to a dataset root.
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
EVICT_TO_FRACTION = 0.9
TOUCH_AFTER_NS = 15 * 60 * 1_000_000_000


class CachedVideoSource:
    def __init__(self, drive, cache_dir, max_bytes: int = 2_000_000_000):
        self._drive = drive
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = max_bytes
        self._evict_lock = threading.Lock()
        self._locks: dict = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def _lock_for(self, file_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[file_id]

    def path(self, file_id: str) -> Path:
        if not isinstance(file_id, str) or not _FILE_ID.fullmatch(file_id):
            raise ValueError("invalid drive file id")
        target = self._dir / f"{file_id}.mp4"
        with self._lock_for(file_id):
            if not target.is_file():
                self._store(target, self._drive.download(file_id))
                self._evict(keep=target)
        # Mark as recently used, but rarely: the mtime is also the ETag /
        # Last-Modified the browser's range requests revalidate against.
        now = time.time_ns()
        if now - target.stat().st_mtime_ns > TOUCH_AFTER_NS:
            os.utime(target, ns=(now, now))
        return target

    def _store(self, target: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _evict(self, keep: Path) -> None:
        with self._evict_lock:
            entries = []
            for entry in self._dir.glob("*.mp4"):
                try:
                    stat = entry.stat()
                except FileNotFoundError:
                    continue
                entries.append((stat.st_mtime_ns, stat.st_size, entry))
            total = sum(size for _, size, _ in entries)
            if total <= self._max:
                return
            for _, size, entry in sorted(entries):
                if entry == keep:
                    continue
                entry.unlink(missing_ok=True)
                total -= size
                if total <= self._max * EVICT_TO_FRACTION:
                    break


class LocalVideoSource:
    def __init__(self, root):
        self._root = Path(root).resolve()

    def path(self, ref: str) -> Path:
        target = (self._root / ref).resolve()
        if self._root not in target.parents or not target.is_file():
            raise ValueError("video outside the dataset root or missing")
        return target
