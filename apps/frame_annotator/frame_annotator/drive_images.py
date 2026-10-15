"""Image source for the web app: Drive sheets behind a size-capped disk cache."""
from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from pathlib import Path

_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
EVICT_TO_FRACTION = 0.9


class CachedImageSource:
    def __init__(self, drive, cache_dir, max_bytes: int = 2_000_000_000):
        self._drive = drive
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = max_bytes
        self._lock = threading.Lock()

    def __call__(self, file_id: str) -> bytes:
        if not isinstance(file_id, str) or not _FILE_ID.match(file_id):
            raise ValueError("invalid drive file id")
        path = self._dir / f"{file_id}.jpg"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            data = self._drive.download(file_id)
            self._store(path, data)
            return data
        now = time.time_ns()
        os.utime(path, ns=(now, now))  # mark as recently used
        return data

    def _store(self, path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
            now = time.time_ns()
            os.utime(path, ns=(now, now))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._evict()

    def _evict(self) -> None:
        with self._lock:
            entries = []
            for entry in self._dir.glob("*.jpg"):
                try:
                    stat = entry.stat()
                except FileNotFoundError:
                    continue
                entries.append((stat.st_mtime_ns, stat.st_size, entry))
            total = sum(size for _, size, _ in entries)
            if total <= self._max:
                return
            for _, size, entry in sorted(entries):
                entry.unlink(missing_ok=True)
                total -= size
                if total <= self._max * EVICT_TO_FRACTION:
                    break
