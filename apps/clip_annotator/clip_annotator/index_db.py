"""Links-only SQLite index: clip -> one full-recording video.

Holds no labels. ``video_ref`` is the Drive file id of the clip's ``video.mp4``
(production) or its path relative to the dataset root (local development).
Built once by the publisher, then read-only at serve time.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .events import is_safe_id

MAX_DURATION_S = 3600.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    clip_id     TEXT PRIMARY KEY,
    location    TEXT NOT NULL,
    recording   TEXT NOT NULL,
    duration_s  REAL NOT NULL CHECK (duration_s > 0),
    fps         REAL NOT NULL CHECK (fps > 0),
    frame_count INTEGER NOT NULL CHECK (frame_count > 0),
    video_ref   TEXT NOT NULL,
    video_bytes INTEGER NOT NULL CHECK (video_bytes > 0)
);
"""


@dataclass(frozen=True)
class Clip:
    clip_id: str
    location: str
    recording: str
    duration_s: float
    fps: float
    frame_count: int
    video_ref: str
    video_bytes: int


_COLUMNS = "clip_id, location, recording, duration_s, fps, frame_count, video_ref, video_bytes"


class IndexDb:
    def __init__(self, path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def create(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.executescript(_SCHEMA)

    def add_clip(self, clip: Clip) -> None:
        if not is_safe_id(clip.clip_id):
            raise ValueError(f"unsafe clip id {clip.clip_id!r}")
        if not 0 < clip.duration_s <= MAX_DURATION_S:
            raise ValueError(f"duration_s must be in (0, {MAX_DURATION_S}]")
        try:
            with closing(self._connect()) as conn:
                with conn:
                    conn.execute(f"INSERT INTO clips ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                 (clip.clip_id, clip.location, clip.recording, clip.duration_s,
                                  clip.fps, clip.frame_count, clip.video_ref, clip.video_bytes))
        except sqlite3.IntegrityError as error:
            raise ValueError(f"clip {clip.clip_id!r} could not be added: {error}") from error

    def clips(self) -> list:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM clips ORDER BY location, recording, clip_id").fetchall()
        return [Clip(*row) for row in rows]

    def clip(self, clip_id: str):
        with closing(self._connect()) as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM clips WHERE clip_id = ?",
                               (clip_id,)).fetchone()
        return Clip(*row) if row else None

    def clip_ids(self) -> set:
        with closing(self._connect()) as conn:
            return {row[0] for row in conn.execute("SELECT clip_id FROM clips")}
