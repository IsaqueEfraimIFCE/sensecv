"""Per-annotator event storage.

``LocalDirStore`` is the development/test backend; ``DriveEventStore`` (in
``drive_store``) implements the same interface. Each annotator's work lives
under its own folder, so annotators never see or overwrite one another.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .events import ClipEvents, is_safe_id
from .index_db import Clip

log = logging.getLogger(__name__)


class EventStore(Protocol):
    def load(self, annotator_id: str, clip: Clip) -> ClipEvents: ...

    def save(self, annotator_id: str, clip_id: str, doc: ClipEvents) -> None: ...

    def summaries(self, annotator_id: str) -> dict: ...


def _check_id(value: str, kind: str) -> str:
    if not is_safe_id(value):
        raise ValueError(f"invalid {kind} id")
    return value


def encode_doc(clip_id: str, doc: ClipEvents) -> bytes:
    """Serialized form shared by every backend (local files, Drive files)."""
    payload = {
        "clip_id": clip_id,
        "duration_s": doc.duration_s,
        "fps": doc.fps,
        **doc.to_json(),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def decode_doc(raw: bytes, clip: Clip) -> ClipEvents:
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("stored annotation must be a JSON object")
    if data.get("clip_id") != clip.clip_id:
        raise ValueError(f"stored annotation belongs to {data.get('clip_id')!r}, not {clip.clip_id!r}")
    return ClipEvents.from_json(data, clip.duration_s, clip.fps)


def summary_of(raw: bytes) -> dict:
    """List-view summary of a stored file, without the index (trusts its header)."""
    data = json.loads(raw.decode("utf-8"))
    return ClipEvents.from_json(data, float(data["duration_s"]), float(data["fps"])).summary()


class LocalDirStore:
    def __init__(self, root):
        self._root = Path(root)

    def _folder(self, annotator_id: str) -> Path:
        return self._root / _check_id(annotator_id, "annotator")

    def _file(self, annotator_id: str, clip_id: str) -> Path:
        return self._folder(annotator_id) / f"{_check_id(clip_id, 'clip')}.json"

    def load(self, annotator_id: str, clip: Clip) -> ClipEvents:
        path = self._file(annotator_id, clip.clip_id)
        if not path.is_file():
            return ClipEvents.empty(clip.duration_s, clip.fps)
        return decode_doc(path.read_bytes(), clip)

    def save(self, annotator_id: str, clip_id: str, doc: ClipEvents) -> None:
        path = self._file(annotator_id, clip_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = encode_doc(clip_id, doc)
        # unique temp name: concurrent writers must never share a half-written file
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def summaries(self, annotator_id: str) -> dict:
        folder = self._folder(annotator_id)
        if not folder.is_dir():
            return {}
        result = {}
        for path in sorted(folder.glob("*.json")):
            try:
                result[path.stem] = summary_of(path.read_bytes())
            except (ValueError, KeyError, TypeError, OSError):
                log.warning("skipping unreadable annotation file %s", path)
        return result
