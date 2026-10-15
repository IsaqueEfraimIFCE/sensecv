"""Sprite sheets: many frames packed into one JPEG for cheap transport.

Labels stay per frame. A sheet is always ``cols`` tiles wide, so the browser can
address any frame with CSS percentages; only the last sheet is shorter.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

DEFAULT_TILE = 256
DEFAULT_COLS = 8
DEFAULT_ROWS = 5
DEFAULT_QUALITY = 80


@dataclass(frozen=True)
class SheetImage:
    sheet_index: int
    first_frame: int
    frame_count: int
    cols: int
    tile_w: int
    tile_h: int
    jpeg: bytes


def _tile(path: Path, size: int) -> Image.Image:
    with Image.open(path) as frame:
        frame.draft("RGB", (size, size))  # decode JPEGs at reduced scale: much faster
        return ImageOps.fit(frame.convert("RGB"), (size, size), method=Image.Resampling.BILINEAR)


def build_sheets(frame_paths, *, tile=DEFAULT_TILE, cols=DEFAULT_COLS, rows=DEFAULT_ROWS,
                 quality=DEFAULT_QUALITY) -> list:
    """Pack ``frame_paths`` (in order) into JPEG sheets of at most cols*rows frames."""
    frame_paths = list(frame_paths)
    if not frame_paths:
        raise ValueError("a clip needs at least one frame")
    per_sheet = cols * rows
    sheets = []
    for sheet_index, start in enumerate(range(0, len(frame_paths), per_sheet)):
        chunk = frame_paths[start:start + per_sheet]
        sheet_rows = -(-len(chunk) // cols)
        canvas = Image.new("RGB", (cols * tile, sheet_rows * tile))
        for k, path in enumerate(chunk):
            canvas.paste(_tile(path, tile), ((k % cols) * tile, (k // cols) * tile))
        buffer = io.BytesIO()
        canvas.save(buffer, "JPEG", quality=quality, optimize=True)
        sheets.append(SheetImage(sheet_index, start, len(chunk), cols, tile, tile, buffer.getvalue()))
    return sheets
