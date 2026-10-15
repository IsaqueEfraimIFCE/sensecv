"""Per-annotator annotation storage.

``LocalDirStore`` is the development/test backend. A Google Drive backend must
implement the same ``AnnotationStore`` interface. Each annotator's work lives
under its own key, so annotators never see or overwrite one another.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .labels import ClipAnnotation, is_safe_id

log = logging.getLogger(__name__)


class AnnotationStore(Protocol):
    def load(self, annotator_id: str, clip_id: str, frame_count: int) -> ClipAnnotation: ...

    def save(self, annotator_id: str, clip_id: str, annotation: ClipAnnotation) -> None: ...

    def labelled_counts(self, annotator_id: str) -> dict: ...


def _check_id(value: str, kind: str) -> str:
    if not is_safe_id(value):
        raise ValueError(f"invalid {kind} id")
    return value


class LocalDirStore:
    def __init__(self, root):
        self._root = Path(root)

    def _folder(self, annotator_id: str) -> Path:
        return self._root / _check_id(annotator_id, "annotator")

    def _file(self, annotator_id: str, clip_id: str) -> Path:
        return self._folder(annotator_id) / f"{_check_id(clip_id, 'clip')}.json"

    def load(self, annotator_id: str, clip_id: str, frame_count: int) -> ClipAnnotation:
        path = self._file(annotator_id, clip_id)
        if not path.is_file():
            return ClipAnnotation.empty(frame_count)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["frame_count"] != frame_count:
            raise ValueError(
                f"stored annotation for {clip_id!r} has {data['frame_count']} frames, "
                f"index says {frame_count}"
            )
        return ClipAnnotation.from_runs(frame_count, data["runs"])

    def save(self, annotator_id: str, clip_id: str, annotation: ClipAnnotation) -> None:
        path = self._file(annotator_id, clip_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clip_id": clip_id,
            "frame_count": annotation.frame_count,
            "runs": annotation.to_runs(),
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        # unique temp name: concurrent writers must never share a half-written file
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def labelled_counts(self, annotator_id: str) -> dict:
        folder = self._folder(annotator_id)
        if not folder.is_dir():
            return {}
        counts = {}
        for path in sorted(folder.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                clip = ClipAnnotation.from_runs(data["frame_count"], data["runs"])
            except (ValueError, KeyError, TypeError, OSError):
                log.warning("skipping unreadable annotation file %s", path)
                continue
            counts[path.stem] = clip.labelled_count
        return counts
