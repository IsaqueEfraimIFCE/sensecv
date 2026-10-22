"""Google Drive event store (same design as the frame annotator's).

Layout under the annotations folder::

    <annotator>/<clip>.json      the clip's events (see ``store.encode_doc``)
    <annotator>/_progress.json   {clip: summary}, so listing costs 1 read

Single-writer design: one server process owns the state. With
``write_behind=True`` (production) ``save`` only updates memory and a
background thread coalesces and uploads to Drive, so an edit never waits for
the Drive API (3-10 s). Unsaved work stays pinned in memory, is retried until
it succeeds, and is flushed by ``close`` at shutdown. Run one gunicorn worker.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict

from .events import ClipEvents
from .index_db import Clip
from .store import _check_id, decode_doc, encode_doc, summary_of

log = logging.getLogger(__name__)

PROGRESS_NAME = "_progress.json"
JSON_MIME = "application/json"
STALE_AFTER_SEC = 120.0  # unsynced work older than this makes health() fail
_BAD_REMOTE = (ValueError, KeyError, TypeError, AttributeError)


class DriveEventStore:
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
        self._cache: OrderedDict = OrderedDict()  # (annotator, clip) -> ClipEvents
        self._progress: dict = {}     # annotator -> {"id": file id | None, "summaries": dict}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._unsaved: dict = {}      # key -> newest doc not yet confirmed on Drive
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

    def _remember_locked(self, key, doc: ClipEvents) -> None:
        self._cache[key] = doc
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    # -- progress ----------------------------------------------------------
    def _rebuild_summaries(self, folder: str) -> dict:
        result = {}
        for name, file_id in self._drive.list_names(folder).items():
            if name == PROGRESS_NAME or not name.endswith(".json"):
                continue
            try:
                result[name[:-5]] = summary_of(self._drive.download(file_id))
            except _BAD_REMOTE:
                log.warning("skipping unreadable annotation file %s", name)
        return result

    def _progress_state(self, annotator_id: str, folder: str) -> dict:
        with self._lock:
            state = self._progress.get(annotator_id)
        if state is not None:
            return state
        progress_id = self._drive.find(PROGRESS_NAME, folder)
        summaries = None
        if progress_id:
            try:
                raw = json.loads(self._drive.download(progress_id).decode("utf-8"))
                if not isinstance(raw, dict) or not all(isinstance(v, dict) for v in raw.values()):
                    raise ValueError("progress file is not {clip: summary}")
                summaries = {str(k): dict(v) for k, v in raw.items()}
            except _BAD_REMOTE:
                log.warning("progress file for %s is unreadable; rebuilding", annotator_id)
        if summaries is None:
            summaries = self._rebuild_summaries(folder)
        state = {"id": progress_id, "summaries": summaries}
        with self._lock:
            # Another thread may have read it meanwhile and already recorded a
            # save into its copy: keep that one, never replace it.
            return self._progress.setdefault(annotator_id, state)

    # -- Drive writes (used inline or by the writer thread) ----------------
    def _persist(self, key, doc: ClipEvents) -> None:
        annotator_id, clip_id = key
        folder = self._folder(annotator_id, create=True)
        with self._lock:
            known = self._clip_files.get(key)
        file_id = self._drive.upload(
            f"{clip_id}.json", encode_doc(clip_id, doc), folder, JSON_MIME, file_id=known)
        with self._lock:
            self._clip_files[key] = file_id
        self._progress_state(annotator_id, folder)["summaries"][clip_id] = doc.summary()

    def _write_progress(self, annotator_id: str) -> None:
        folder = self._folder(annotator_id, create=True)
        state = self._progress_state(annotator_id, folder)
        state["id"] = self._drive.upload(
            PROGRESS_NAME, json.dumps(state["summaries"]).encode("utf-8"), folder, JSON_MIME,
            file_id=state["id"],
        )

    # -- EventStore interface ---------------------------------------------
    def load(self, annotator_id: str, clip: Clip) -> ClipEvents:
        _check_id(annotator_id, "annotator")
        _check_id(clip.clip_id, "clip")
        key = (annotator_id, clip.clip_id)
        with self._lock:
            cached = self._unsaved.get(key) or self._cache.get(key)
        if cached is not None:
            return cached
        folder = self._folder(annotator_id, create=False)
        if folder is None:
            return ClipEvents.empty(clip.duration_s, clip.fps)
        with self._lock:
            file_id = self._clip_files.get(key)
        file_id = file_id or self._drive.find(f"{clip.clip_id}.json", folder)
        if file_id is None:
            return ClipEvents.empty(clip.duration_s, clip.fps)
        doc = decode_doc(self._drive.download(file_id), clip)
        with self._lock:
            self._clip_files[key] = file_id
            # A save that landed during the download is newer than what we read.
            newer = self._unsaved.get(key) or self._cache.get(key)
            if newer is not None:
                return newer
            self._remember_locked(key, doc)
        return doc

    def save(self, annotator_id: str, clip_id: str, doc: ClipEvents) -> None:
        _check_id(annotator_id, "annotator")
        _check_id(clip_id, "clip")
        key = (annotator_id, clip_id)
        if not self._write_behind:
            self._persist(key, doc)
            self._write_progress(annotator_id)
            with self._lock:
                self._remember_locked(key, doc)
            return
        with self._cond:
            if not self._unsaved:
                self._last_success = time.monotonic()  # the stale clock starts now
            self._unsaved[key] = doc
            self._queue.add(key)
            self._remember_locked(key, doc)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="drive-writer", daemon=True)
                self._thread.start()
            self._cond.notify_all()

    def summaries(self, annotator_id: str) -> dict:
        _check_id(annotator_id, "annotator")
        result: dict = {}
        folder = self._folder(annotator_id, create=False)
        if folder is not None:
            result = dict(self._progress_state(annotator_id, folder)["summaries"])
        with self._lock:
            for (annotator, clip), doc in self._unsaved.items():
                if annotator == annotator_id:
                    result[clip] = doc.summary()
        return result

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
            for key, doc in snapshots.items():
                try:
                    self._persist(key, doc)
                except Exception:  # noqa: BLE001 - keep the work pinned and retry
                    failed = True
                    log.exception("could not save %s/%s; will retry", *key)
                    with self._cond:
                        self._queue.add(key)
                    continue
                annotators.add(key[0])
                with self._cond:
                    if self._unsaved.get(key) is doc:
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
