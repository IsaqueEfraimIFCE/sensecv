"""Links-only SQLite index: clip -> Drive sprite sheets.

Holds no labels and no images. Annotations live in the annotation store and the
images live on Google Drive; this file only says which Drive file holds which
frames. Built once, then read-only at serve time.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .labels import is_safe_id

MAX_FRAMES_PER_CLIP = 20_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    clip_id     TEXT PRIMARY KEY,
    location    TEXT NOT NULL,
    frame_count INTEGER NOT NULL CHECK (frame_count > 0)
);
CREATE TABLE IF NOT EXISTS sheets (
    clip_id       TEXT NOT NULL REFERENCES clips (clip_id),
    sheet_index   INTEGER NOT NULL,
    drive_file_id TEXT NOT NULL,
    first_frame   INTEGER NOT NULL,
    frame_count   INTEGER NOT NULL CHECK (frame_count > 0),
    cols          INTEGER NOT NULL CHECK (cols > 0),
    tile_w        INTEGER NOT NULL CHECK (tile_w > 0),
    tile_h        INTEGER NOT NULL CHECK (tile_h > 0),
    PRIMARY KEY (clip_id, sheet_index)
);
"""


@dataclass(frozen=True)
class Clip:
    clip_id: str
    location: str
    frame_count: int


@dataclass(frozen=True)
class Sheet:
    clip_id: str
    sheet_index: int
    drive_file_id: str
    first_frame: int
    frame_count: int
    cols: int
    tile_w: int
    tile_h: int


class IndexDb:
    def __init__(self, path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def create(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.executescript(_SCHEMA)

    def add_clip(self, clip: Clip, sheets: list) -> None:
        """Insert a clip with its sheets; the sheets must tile frames 0..N-1."""
        if not is_safe_id(clip.clip_id):
            raise ValueError(f"unsafe clip id {clip.clip_id!r}")
        if not 0 < clip.frame_count <= MAX_FRAMES_PER_CLIP:
            raise ValueError(f"frame_count must be 1..{MAX_FRAMES_PER_CLIP}")
        expected = 0
        for sheet in sorted(sheets, key=lambda s: s.sheet_index):
            if sheet.clip_id != clip.clip_id:
                raise ValueError(f"sheet belongs to {sheet.clip_id!r}, not {clip.clip_id!r}")
            if sheet.first_frame != expected:
                raise ValueError(f"sheets leave a gap or overlap at frame {expected}")
            expected += sheet.frame_count
        if expected != clip.frame_count:
            raise ValueError(f"sheets cover {expected} frames, clip has {clip.frame_count}")
        try:
            with closing(self._connect()) as conn:
                with conn:
                    conn.execute(
                        "INSERT INTO clips VALUES (?, ?, ?)",
                        (clip.clip_id, clip.location, clip.frame_count),
                    )
                    conn.executemany(
                        "INSERT INTO sheets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (s.clip_id, s.sheet_index, s.drive_file_id, s.first_frame,
                             s.frame_count, s.cols, s.tile_w, s.tile_h)
                            for s in sheets
                        ],
                    )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"clip {clip.clip_id!r} could not be added: {error}") from error

    def clips(self) -> list:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT clip_id, location, frame_count FROM clips ORDER BY location, clip_id"
            ).fetchall()
        return [Clip(*row) for row in rows]

    def clip(self, clip_id: str):
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT clip_id, location, frame_count FROM clips WHERE clip_id = ?", (clip_id,)
            ).fetchone()
        return Clip(*row) if row else None

    def sheets(self, clip_id: str) -> list:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT clip_id, sheet_index, drive_file_id, first_frame, frame_count, cols,"
                " tile_w, tile_h FROM sheets WHERE clip_id = ? ORDER BY sheet_index",
                (clip_id,),
            ).fetchall()
        return [Sheet(*row) for row in rows]
