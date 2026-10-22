"""Read an MP4's playable duration from its ``moov/mvhd`` header (no ffmpeg).

The phone app's ``metadata.json`` reports ~0.7 s more than the video the
browser plays, so the publisher uses this as the clip's timeline length.
"""
from __future__ import annotations

import os
import struct
from typing import BinaryIO, Iterator, Optional, Tuple

_HEADER = struct.Struct(">I4s")


def _boxes(handle: BinaryIO, end: int) -> Iterator[Tuple[bytes, int, int]]:
    """Yield ``(type, body_start, box_end)`` for the boxes up to ``end``."""
    while handle.tell() + _HEADER.size <= end:
        start = handle.tell()
        size, kind = _HEADER.unpack(handle.read(_HEADER.size))
        if size == 1:
            size = struct.unpack(">Q", handle.read(8))[0]
        elif size == 0:
            size = end - start
        if size < _HEADER.size or start + size > end:
            return  # truncated or corrupt: stop instead of looping
        yield kind, handle.tell(), start + size
        handle.seek(start + size)


def mp4_duration(path) -> Optional[float]:
    """Duration in seconds, or None when the file has no readable ``mvhd``."""
    try:
        with open(path, "rb") as handle:
            for kind, body, stop in _boxes(handle, os.path.getsize(path)):
                if kind != b"moov":
                    continue
                handle.seek(body)
                for inner, inner_body, _ in _boxes(handle, stop):
                    if inner != b"mvhd":
                        continue
                    handle.seek(inner_body)
                    version = handle.read(4)[0]
                    if version == 1:
                        handle.read(16)
                        timescale, duration = struct.unpack(">IQ", handle.read(12))
                    else:
                        handle.read(8)
                        timescale, duration = struct.unpack(">II", handle.read(8))
                    return duration / timescale if timescale and duration else None
    except (OSError, struct.error, IndexError):
        return None
    return None
