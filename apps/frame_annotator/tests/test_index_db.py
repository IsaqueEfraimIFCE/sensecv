"""The SQLite index holds only links: clip -> Drive sprite sheets."""
import pytest

from frame_annotator.index_db import Clip, IndexDb, Sheet


def _sheet(index, first, count, file_id=None):
    return Sheet("c1", index, file_id or f"drive-{index}", first, count, 10, 160, 120)


@pytest.fixture
def db(tmp_path):
    database = IndexDb(tmp_path / "index.sqlite")
    database.create()
    return database


def test_clip_and_sheets_roundtrip(db):
    db.add_clip(Clip("c1", "E4_UFC_Benfica_corridors", 25), [_sheet(0, 0, 20), _sheet(1, 20, 5)])

    assert db.clips() == [Clip("c1", "E4_UFC_Benfica_corridors", 25)]
    assert db.clip("c1").frame_count == 25
    assert [s.drive_file_id for s in db.sheets("c1")] == ["drive-0", "drive-1"]


def test_unknown_clip_is_none(db):
    assert db.clip("missing") is None
    assert db.sheets("missing") == []


def test_sheets_must_cover_every_frame_without_gaps(db):
    with pytest.raises(ValueError):
        db.add_clip(Clip("c1", "L", 25), [_sheet(0, 0, 20)])  # frames 20..24 missing
    with pytest.raises(ValueError):
        db.add_clip(Clip("c1", "L", 25), [_sheet(0, 0, 20), _sheet(1, 21, 4)])  # gap


def test_failed_add_leaves_nothing_behind(db):
    with pytest.raises(ValueError):
        db.add_clip(Clip("c1", "L", 25), [_sheet(0, 0, 20)])

    assert db.clips() == []


def test_duplicate_clip_is_rejected(db):
    db.add_clip(Clip("c1", "L", 5), [_sheet(0, 0, 5)])

    with pytest.raises(ValueError):
        db.add_clip(Clip("c1", "L", 5), [_sheet(0, 0, 5)])


def test_absurd_frame_counts_are_rejected(db):
    with pytest.raises(ValueError):
        db.add_clip(Clip("c1", "L", 50_000), [Sheet("c1", 0, "d", 0, 50_000, 10, 8, 8)])


def test_clip_ids_must_be_url_and_path_safe(db):
    with pytest.raises(ValueError):
        db.add_clip(Clip('bad"id', "L", 5), [Sheet('bad"id', 0, "d", 0, 5, 5, 8, 8)])


def test_schema_has_no_label_columns(db):
    import sqlite3

    with sqlite3.connect(db.path) as conn:
        columns = {
            row[1]
            for table in ("clips", "sheets")
            for row in conn.execute(f"PRAGMA table_info({table})")
        }

    assert not {"obstacle", "deviation", "unusable", "label"} & columns
