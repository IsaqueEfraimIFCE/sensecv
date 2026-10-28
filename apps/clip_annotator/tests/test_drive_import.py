import os
import zipfile

from clip_annotator.drive_import import extract_recordings, run_import
from clip_annotator.index_db import IndexDb
from conftest import FAKE_MP4, write_recording


def make_collection_zip(tmp_path, collection="SenseCV-06-06-2026-IFCE"):
    """A zip shaped like the Drive collections: <collection>/<recording>/files."""
    src = tmp_path / "src"
    for recording in ("01", "02"):
        write_recording(src / collection / recording)
        (src / collection / recording / "accelerations.json").write_text("[]", encoding="utf-8")
    path = tmp_path / f"{collection}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file in sorted(src.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(src).as_posix())
        archive.writestr(f"__MACOSX/{collection}/01/._video.mp4", b"junk")
        archive.writestr("../escape/video.mp4", b"evil")
    return path


def test_extract_keeps_only_what_the_annotator_needs(tmp_path):
    zip_path = make_collection_zip(tmp_path)
    dest = tmp_path / "raw" / "SenseCV-06-06-2026-IFCE"
    assert extract_recordings(zip_path, dest) == 6  # 2 recordings x (mp4, frames, metadata)
    recording = dest / "SenseCV-06-06-2026-IFCE" / "01"
    assert sorted(p.name for p in recording.iterdir()) == ["frames.json", "metadata.json", "video.mp4"]
    assert not (tmp_path / "raw" / "escape").exists()
    assert not (dest / "__MACOSX").exists()


def test_run_import_indexes_with_hard_links_and_resumes(tmp_path):
    zip_path = make_collection_zip(tmp_path)
    calls = []

    def download(file_id, path):
        calls.append(file_id)
        path.write_bytes(zip_path.read_bytes())

    zips = [{"id": "fid1", "name": zip_path.name, "size": zip_path.stat().st_size}]
    report = run_import(zips, tmp_path / "data" / "raw", tmp_path / "data", download)
    assert report["imported"] == [("SenseCV-06-06-2026-IFCE", 6)]
    assert report["indexed"] == 2 and report["failed"] == []
    clip = IndexDb(tmp_path / "data" / "index.sqlite").clip("SenseCV-06-06-2026-IFCE__01")
    video = tmp_path / "data" / "videos" / clip.video_ref
    assert video.read_bytes() == FAKE_MP4
    assert os.stat(video).st_nlink == 2  # hard link: no second copy on disk
    assert not list((tmp_path / "data" / "raw" / "_zips").iterdir())  # zip deleted

    again = run_import(zips, tmp_path / "data" / "raw", tmp_path / "data", download)
    assert again["skipped"] == ["SenseCV-06-06-2026-IFCE"] and again["indexed"] == 0
    assert calls == ["fid1"]


def test_run_import_reports_a_failed_download_and_continues(tmp_path):
    def broken(file_id, path):
        path.write_bytes(b"partial")
        raise ConnectionError("drive dropped the connection")

    report = run_import([{"id": "x", "name": "SenseCV-Bad.zip"}], tmp_path / "raw", tmp_path / "data", broken)
    assert report["failed"][0][0] == "SenseCV-Bad"
    assert not (tmp_path / "raw" / "_zips" / "SenseCV-Bad.zip").exists()
    assert not (tmp_path / "raw" / "SenseCV-Bad" / ".complete").exists()
