"""Publisher: manifest -> per-clip sprite sheets on Drive + links-only index."""
import csv

import pytest

from frame_annotator.index_db import IndexDb
from frame_annotator.publish import jobs_from_manifest, publish_all

SHEETS = "sheets-folder"


def test_manifest_is_grouped_into_one_job_per_clip(tiny_dataset):
    root, manifest = tiny_dataset

    jobs = jobs_from_manifest(root, manifest)

    assert [(j.clip_id, j.location, len(j.frames)) for j in jobs] == [
        ("E1_0207_01", "E1_IFCE_corridors", 3),
        ("E1_0207_02", "E1_IFCE_corridors", 2),
    ]
    assert jobs[0].frames[0].name == "0207_01_0000.jpg"


def test_gaps_in_frame_indices_are_rejected(tiny_dataset, tmp_path):
    root, manifest = tiny_dataset
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    broken = tmp_path / "broken.csv"
    with broken.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([r for r in rows if not (r["image"] == "0207_01_0001.jpg")])

    with pytest.raises(ValueError):
        jobs_from_manifest(root, broken)


def test_publish_uploads_sheets_and_records_only_links(tiny_dataset, tmp_path, drive):
    root, manifest = tiny_dataset
    index = IndexDb(tmp_path / "index.sqlite")
    index.create()

    summary = publish_all(jobs_from_manifest(root, manifest), drive, SHEETS, index,
                          tile=16, cols=2, rows=1, workers=2)

    assert (summary.published, summary.skipped, summary.failed) == (2, 0, [])
    assert [c.frame_count for c in index.clips()] == [3, 2]
    sheets = index.sheets("E1_0207_01")  # 3 frames, 2 per sheet -> 2 sheets
    assert [(s.first_frame, s.frame_count) for s in sheets] == [(0, 2), (2, 1)]
    names = drive.list_names(SHEETS)
    assert set(names) == {"E1_0207_01__s00.jpg", "E1_0207_01__s01.jpg", "E1_0207_02__s00.jpg"}
    assert sheets[0].drive_file_id == names["E1_0207_01__s00.jpg"]


def test_rerun_skips_published_clips(tiny_dataset, tmp_path, drive):
    root, manifest = tiny_dataset
    index = IndexDb(tmp_path / "index.sqlite")
    index.create()
    jobs = jobs_from_manifest(root, manifest)
    publish_all(jobs, drive, SHEETS, index, tile=16, cols=2, rows=1)
    uploads = drive.uploads

    summary = publish_all(jobs, drive, SHEETS, index, tile=16, cols=2, rows=1)

    assert (summary.published, summary.skipped) == (0, 2)
    assert drive.uploads == uploads


def test_a_failing_clip_is_reported_and_the_rest_continue(tiny_dataset, tmp_path, drive):
    root, manifest = tiny_dataset
    (root / "E1_IFCE_corridors" / "0207_01_0001.jpg").write_bytes(b"not an image")
    index = IndexDb(tmp_path / "index.sqlite")
    index.create()

    summary = publish_all(jobs_from_manifest(root, manifest), drive, SHEETS, index,
                          tile=16, cols=2, rows=1)

    assert summary.published == 1
    assert [clip_id for clip_id, _ in summary.failed] == ["E1_0207_01"]
    assert [c.clip_id for c in index.clips()] == ["E1_0207_02"]
