"""Shared fixtures: an in-memory Drive double and a tiny on-disk dataset."""
import csv
import itertools

import pytest
from PIL import Image


class FakeDrive:
    """Implements the DriveClient interface in memory and counts calls."""

    def __init__(self):
        self._ids = itertools.count(1)
        self.files = {}  # id -> {"name", "parent", "data", "mime"}
        self.downloads = 0
        self.uploads = 0

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


@pytest.fixture
def tiny_dataset(tmp_path):
    """Two clips (3 and 2 frames) of 40x40 JPEGs plus a manifest.csv."""
    root = tmp_path / "dataset"
    folder = root / "E1_IFCE_corridors"
    folder.mkdir(parents=True)
    rows = []
    for prefix, count in [("0207_01", 3), ("0207_02", 2)]:
        for i in range(count):
            name = f"{prefix}_{i:04d}.jpg"
            Image.new("RGB", (40, 40), (i * 60, 40, 200)).save(folder / name, "JPEG")
            rows.append({
                "env": "E1", "env_name": "IFCE_corridors", "frame_index": i,
                "path": f"E1_IFCE_corridors/{name}", "image": name,
            })
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root, manifest
