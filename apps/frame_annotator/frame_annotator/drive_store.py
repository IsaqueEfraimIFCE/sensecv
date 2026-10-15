"""Google Drive annotation store.

Layout under the annotations folder::

    <annotator>/<clip>.json      one run-length annotation per clip
    <annotator>/_progress.json   {clip: labelled_count}, so listing costs 1 read

Single-writer design: one server process owns the state. With
``write_behind=True`` (production) ``save`` only updates memory and a
background thread coalesces and uploads to Drive, so a label action never waits
for the Drive API (3-10 s). Unsaved work stays pinned in memory, is retried
until it succeeds, and is flushed by ``close`` at shutdown. Run the app with one
gunicorn worker.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict

from .labels import ClipAnnotation
from .store import _check_id, decode_annotation, encode_annotation, labelled_count_of

log = logging.getLogger(__name__)

PROGRESS_NAME = "_progress.json"
JSON_MIME = "application/json"
STALE_AFTER_SEC = 120.0  # unsynced work older than this makes health() fail
_BAD_REMOTE = (ValueError, KeyError, TypeError, AttributeError)


class DriveAnnotationStore:
    def __init__(self, drive, root_folder_id: str, cache_size: int = 256, *,
                 write_behind: bool = False, coalesce_sec: float = 0.25, retry_sec: float = 2.0):
        self._drive = drive
        self._root = root_folder_id
        self._cache_size = cache_size
        self._write_behind = write_behind
        self._coalesce = coalesce_sec
        self._retry = retry_sec
        self._folders: dict = {}      # annotator -> folder id
        self._clip_files: dict = {}   # (annotator, clip) -> file id
        self._cache: OrderedDict = OrderedDict()  # (annotator, clip) -> ClipAnnotation
        self._progress: dict = {}     # annotator -> {"id": file id | None, "counts": dict}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._unsaved: dict = {}      # key -> newest annotation not yet confirmed on Drive
        self._queue: set = set()      # keys waiting for the writer
        self._progress_dirty: set = set()
        self._busy = False
        self._closed = False
        self._thread = None
        self._last_success = time.monotonic()

    # -- folders / cache ---------------------------------------------------
    def _folder(self, annotator_id: str, *, create: bool):
        with self._lock:
            known = self._folders.get(annotator_id)
        if known:
            return known
        lookup = self._drive.ensure_folder if create else self._drive.find
        folder = lookup(annotator_id, self._root)
        if folder:
            with self._lock:
                self._folders[annotator_id] = folder
        return folder

    def _remember(self, key, annotation: ClipAnnotation) -> None:
        with self._lock:
            self._cache[key] = annotation
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    # -- progress ----------------------------------------------------------
    def _rebuild_counts(self, folder: str) -> dict:
        counts = {}
        for name, file_id in self._drive.list_names(folder).items():
            if name == PROGRESS_NAME or not name.endswith(".json"):
                continue
            try:
                counts[name[:-5]] = labelled_count_of(self._drive.download(file_id))
            except _BAD_REMOTE:
                log.warning("skipping unreadable annotation file %s", name)
        return counts

    def _progress_state(self, annotator_id: str, folder: str) -> dict:
        with self._lock:
            state = self._progress.get(annotator_id)
        if state is not None:
            return state
        progress_id = self._drive.find(PROGRESS_NAME, folder)
        counts = None
        if progress_id:
            try:
                raw = json.loads(self._drive.download(progress_id).decode("utf-8"))
                counts = {str(k): int(v) for k, v in raw.items()}
            except _BAD_REMOTE:
                log.warning("progress file for %s is unreadable; rebuilding", annotator_id)
        if counts is None:
            counts = self._rebuild_counts(folder)
        state = {"id": progress_id, "counts": counts}
        with self._lock:
            self._progress[annotator_id] = state
        return state

    # -- Drive writes (used inline or by the writer thread) ----------------
    def _persist(self, key, annotation: ClipAnnotation) -> None:
        annotator_id, clip_id = key
        folder = self._folder(annotator_id, create=True)
        with self._lock:
            known = self._clip_files.get(key)
        file_id = self._drive.upload(
            f"{clip_id}.json", encode_annotation(clip_id, annotation), folder, JSON_MIME,
            file_id=known,
        )
        with self._lock:
            self._clip_files[key] = file_id
        self._progress_state(annotator_id, folder)["counts"][clip_id] = annotation.labelled_count

    def _write_progress(self, annotator_id: str) -> None:
        folder = self._folder(annotator_id, create=True)
        state = self._progress_state(annotator_id, folder)
        state["id"] = self._drive.upload(
            PROGRESS_NAME, json.dumps(state["counts"]).encode("utf-8"), folder, JSON_MIME,
            file_id=state["id"],
        )

    # -- AnnotationStore interface ----------------------------------------
    def load(self, annotator_id: str, clip_id: str, frame_count: int) -> ClipAnnotation:
        _check_id(annotator_id, "annotator")
        _check_id(clip_id, "clip")
        key = (annotator_id, clip_id)
        with self._lock:
            cached = self._unsaved.get(key)
            if cached is None:
                cached = self._cache.get(key)
        if cached is not None and cached.frame_count == frame_count:
            return cached
        folder = self._folder(annotator_id, create=False)
        if folder is None:
            return ClipAnnotation.empty(frame_count)
        with self._lock:
            file_id = self._clip_files.get(key)
        file_id = file_id or self._drive.find(f"{clip_id}.json", folder)
        if file_id is None:
            return ClipAnnotation.empty(frame_count)
        annotation = decode_annotation(self._drive.download(file_id), frame_count, clip_id)
        with self._lock:
            self._clip_files[key] = file_id
        self._remember(key, annotation)
        return annotation

    def save(self, annotator_id: str, clip_id: str, annotation: ClipAnnotation) -> None:
        _check_id(annotator_id, "annotator")
        _check_id(clip_id, "clip")
        key = (annotator_id, clip_id)
        if not self._write_behind:
            self._persist(key, annotation)
            self._write_progress(annotator_id)
            self._remember(key, annotation)
            return
        with self._cond:
            if not self._unsaved:
                self._last_success = time.monotonic()  # the stale clock starts now
            self._unsaved[key] = annotation
            self._queue.add(key)
            self._cache[key] = annotation
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="drive-writer", daemon=True)
                self._thread.start()
            self._cond.notify_all()

    def labelled_counts(self, annotator_id: str) -> dict:
        _check_id(annotator_id, "annotator")
        counts: dict = {}
        folder = self._folder(annotator_id, create=False)
        if folder is not None:
            counts = dict(self._progress_state(annotator_id, folder)["counts"])
        with self._lock:
            for (annotator, clip), annotation in self._unsaved.items():
                if annotator == annotator_id:
                    counts[clip] = annotation.labelled_count
        return counts

    # -- write-behind machinery -------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._progress_dirty and not self._closed:
                    self._cond.wait()
                if self._closed and not self._queue and not self._progress_dirty:
                    return
                keys = list(self._queue)
                self._queue.clear()
                snapshots = {k: self._unsaved[k] for k in keys if k in self._unsaved}
                annotators = set(self._progress_dirty)
                self._progress_dirty.clear()
                self._busy = True
            failed = False
            for key, annotation in snapshots.items():
                try:
                    self._persist(key, annotation)
                except Exception:  # noqa: BLE001 - keep the work pinned and retry
                    failed = True
                    log.exception("could not save %s/%s; will retry", *key)
                    with self._cond:
                        self._queue.add(key)
                    continue
                annotators.add(key[0])
                with self._cond:
                    if self._unsaved.get(key) is annotation:
                        del self._unsaved[key]
            for annotator_id in annotators:
                try:
                    self._write_progress(annotator_id)
                except Exception:  # noqa: BLE001
                    failed = True
                    log.exception("could not write progress for %s; will retry", annotator_id)
                    with self._cond:
                        self._progress_dirty.add(annotator_id)
            with self._cond:
                if not failed:
                    self._last_success = time.monotonic()
                self._busy = False
                self._cond.notify_all()
                end = time.monotonic() + (self._retry if failed else self._coalesce)
                while not (self._closed and not failed):
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(remaining)

    def pending_writes(self) -> int:
        with self._lock:
            return len(self._unsaved)

    def flush(self, timeout: float = 30.0) -> bool:
        """Block until everything is on Drive; False if ``timeout`` expires first."""
        if not self._write_behind:
            return True
        deadline = time.monotonic() + timeout
        with self._cond:
            self._cond.notify_all()
            while self._unsaved or self._queue or self._progress_dirty or self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(min(remaining, 0.1))
            return True

    def close(self, timeout: float = 30.0) -> bool:
        """Flush pending work and stop the writer (called at shutdown)."""
        if not self._write_behind:
            return True
        flushed = self.flush(timeout)
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        return flushed

    def health(self) -> dict:
        with self._lock:
            unsynced = len(self._unsaved)
            ok = unsynced == 0 or (time.monotonic() - self._last_success) < STALE_AFTER_SEC
        return {"unsynced": unsynced, "ok": ok}
