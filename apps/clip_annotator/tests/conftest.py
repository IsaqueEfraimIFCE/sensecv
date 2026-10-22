"""Shared fixtures: an in-memory Drive double and a tiny on-disk recording set."""
import itertools
import json
import struct

import pytest

from clip_annotator.index_db import Clip


class FakeDrive:
    """Implements the DriveClient interface in memory and counts calls."""

    def __init__(self):
        self._ids = itertools.count(1)
        self.files = {}  # id -> {"name", "parent", "data", "mime"}
        self.downloads = 0
        self.uploads = 0
        self.fail_uploads = 0  # the next N uploads raise

    def _new(self, name, parent, data, mime):
        file_id = f"fid{next(self._ids)}"
        self.files[file_id] = {"name": name, "parent": parent, "data": data, "mime": mime}
        return file_id

    def find(self, name, parent_id):
        for file_id, f in self.files.items():
            if f["name"] == name and f["parent"] == parent_id:
                return file_id
        return None

    def ensure_folder(self, name, parent_id):
        return self.find(name, parent_id) or self._new(name, parent_id, None, "folder")

    def upload(self, name, data, parent_id, mime="application/octet-stream", file_id=None):
        if self.fail_uploads:
            self.fail_uploads -= 1
            raise ConnectionError("simulated network failure")
        self.uploads += 1
        file_id = file_id or self.find(name, parent_id)
        if file_id is None:
            return self._new(name, parent_id, data, mime)
        self.files[file_id]["data"] = data
        return file_id

    def download(self, file_id):
        self.downloads += 1
        return self.files[file_id]["data"]

    def list_names(self, parent_id):
        return {f["name"]: i for i, f in self.files.items() if f["parent"] == parent_id}


@pytest.fixture
def drive():
    return FakeDrive()


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def mp4_bytes(duration_s=1.2, version=0, timescale=1000) -> bytes:
    """A minimal ISO-BMFF file whose moov/mvhd says ``duration_s``."""
    ticks = round(duration_s * timescale)
    if version == 1:
        times = struct.pack(">QQIQ", 0, 0, timescale, ticks)
    else:
        times = struct.pack(">IIII", 0, 0, timescale, ticks)
    mvhd = box(b"mvhd", bytes([version, 0, 0, 0]) + times + bytes(80))
    return (box(b"ftyp", b"mp42\x00\x00\x00\x00") + box(b"moov", box(b"udta", b"") + mvhd)
            + box(b"mdat", bytes(range(256)) * 40))


FAKE_MP4 = mp4_bytes(1.2)


def write_recording(folder, frames=30, duration=1.2, fps=30, cut=False, metadata=True,
                    header=True):
    folder.mkdir(parents=True)
    (folder / "video.mp4").write_bytes(mp4_bytes(duration) if header else b"\x00" * 64)
    (folder / "frames.json").write_text(json.dumps({"frames": [
        {"frame_id": i, "time_usec": 1_000_000 + int(i * 1e6 / fps)} for i in range(frames)
    ]}), encoding="utf-8")
    if metadata:
        (folder / "metadata.json").write_text(json.dumps(
            {"duration_seconds": duration, "fps_selected": fps}), encoding="utf-8")
    if cut:
        (folder / "cut_info.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    """Two full recordings in one collection, one in another, plus noise."""
    root = tmp_path / "dataset"
    write_recording(root / "SenseCV-01-01-2026-EXEMPLO" / "07")
    write_recording(root / "SenseCV-01-01-2026-EXEMPLO" / "08", frames=60, duration=2.4)
    write_recording(root / "SenseCV-02-01-2026-OUTRO" / "01")
    write_recording(root / "SenseCV-02-01-2026-OUTRO_cuts" / "c1", cut=True)
    write_recording(root / "SenseCV-Teste-Estresse" / "01")
    write_recording(root / "loose" / "nometa", metadata=False)
    return root


@pytest.fixture
def clip():
    return Clip("SenseCV-01-01-2026-EXEMPLO__07", "SenseCV-01-01-2026-EXEMPLO", "07",
                10.0, 30.0, 300, "fid-video", 1234)


def desvio(start=2.0, end=3.0, side="esquerda", obstacle="centro"):
    return {"kind": "desvio", "side": side, "obstacle": obstacle, "start_s": start, "end_s": end}
