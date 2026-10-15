"""Google Drive annotation store.

Layout under the annotations folder::

    <annotator>/<clip>.json      one run-length annotation per clip
    <annotator>/_progress.json   {clip: labelled_count}, so listing costs 1 read

Single-writer design: one server process owns the cache, which is written
through on every save. Run the app with one gunicorn worker.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict

from .labels import ClipAnnotation
from .store import _check_id, decode_annotation, encode_annotation, labelled_count_of

log = logging.getLogger(__name__)

PROGRESS_NAME = "_progress.json"
JSON_MIME = "application/json"
_BAD_REMOTE = (ValueError, KeyError, TypeError, AttributeError)


class DriveAnnotationStore:
    def __init__(self, drive, root_folder_id: str, cache_size: int = 256):
        self._drive = drive
        self._root = root_folder_id
        self._cache_size = cache_size
        self._folders: dict = {}      # annotator -> folder id
        self._clip_files: dict = {}   # (annotator, clip) -> file id
        self._cache: OrderedDict = OrderedDict()  # (annotator, clip) -> ClipAnnotation
        self._progress: dict = {}     # annotator -> {"id": file id | None, "counts": dict}
        self._lock = threading.Lock()

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

    # -- AnnotationStore interface ----------------------------------------
    def load(self, annotator_id: str, clip_id: str, frame_count: int) -> ClipAnnotation:
        _check_id(annotator_id, "annotator")
        _check_id(clip_id, "clip")
        key = (annotator_id, clip_id)
        with self._lock:
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
        folder = self._folder(annotator_id, create=True)
        with self._lock:
            known = self._clip_files.get(key)
        file_id = self._drive.upload(
            f"{clip_id}.json", encode_annotation(clip_id, annotation), folder, JSON_MIME,
            file_id=known,
        )
        with self._lock:
            self._clip_files[key] = file_id
        state = self._progress_state(annotator_id, folder)
        state["counts"][clip_id] = annotation.labelled_count
        state["id"] = self._drive.upload(
            PROGRESS_NAME, json.dumps(state["counts"]).encode("utf-8"), folder, JSON_MIME,
            file_id=state["id"],
        )
        self._remember(key, annotation)

    def labelled_counts(self, annotator_id: str) -> dict:
        _check_id(annotator_id, "annotator")
        folder = self._folder(annotator_id, create=False)
        if folder is None:
            return {}
        return dict(self._progress_state(annotator_id, folder)["counts"])
