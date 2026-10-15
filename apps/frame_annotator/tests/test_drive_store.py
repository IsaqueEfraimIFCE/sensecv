"""Drive-backed annotation store: separate per annotator, resumable, cheap listing."""
import pytest

from frame_annotator.drive_store import PROGRESS_NAME, DriveAnnotationStore
from frame_annotator.labels import ClipAnnotation

ROOT = "annotations-root"


@pytest.fixture
def store(drive):
    return DriveAnnotationStore(drive, ROOT)


def _clip(n=8, upto=3):
    return ClipAnnotation.empty(n).apply(0, upto, obstacle="obstacle", deviation="left")


def test_missing_clip_loads_as_untouched(store):
    assert store.load("ana", "c1", 8) == ClipAnnotation.empty(8)


def test_work_survives_a_new_process(drive, store):
    store.save("ana", "c1", _clip())

    assert DriveAnnotationStore(drive, ROOT).load("ana", "c1", 8) == _clip()


def test_annotators_are_isolated(drive, store):
    store.save("ana", "c1", _clip())

    assert DriveAnnotationStore(drive, ROOT).load("bruno", "c1", 8) == ClipAnnotation.empty(8)
    assert DriveAnnotationStore(drive, ROOT).labelled_counts("bruno") == {}


def test_counts_come_from_one_progress_file_not_every_clip(drive, store):
    store.save("ana", "c1", ClipAnnotation.empty(4).apply(0, 1, obstacle="free_path"))
    store.save("ana", "c2", ClipAnnotation.empty(3).apply(0, 2, unusable=True))
    fresh = DriveAnnotationStore(drive, ROOT)
    before = drive.downloads

    assert fresh.labelled_counts("ana") == {"c1": 2, "c2": 3}
    assert drive.downloads - before == 1


def test_missing_progress_file_is_rebuilt_from_the_clip_files(drive, store):
    store.save("ana", "c1", ClipAnnotation.empty(4).apply(0, 1, obstacle="free_path"))
    folder = drive.find("ana", ROOT)
    del drive.files[drive.find(PROGRESS_NAME, folder)]

    assert DriveAnnotationStore(drive, ROOT).labelled_counts("ana") == {"c1": 2}


def test_saved_work_is_served_from_memory(drive, store):
    store.save("ana", "c1", _clip())
    downloads = drive.downloads

    for _ in range(3):
        assert store.load("ana", "c1", 8) == _clip()

    assert drive.downloads == downloads


def test_frame_count_mismatch_is_an_error(drive, store):
    store.save("ana", "c1", _clip())

    with pytest.raises(ValueError):
        DriveAnnotationStore(drive, ROOT).load("ana", "c1", 9)


def test_corrupt_remote_file_raises_value_error(drive, store):
    store.save("ana", "c1", _clip())
    folder = drive.find("ana", ROOT)
    drive.files[drive.find("c1.json", folder)]["data"] = b"{not json"

    with pytest.raises(ValueError):
        DriveAnnotationStore(drive, ROOT).load("ana", "c1", 8)


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a b"])
def test_ids_are_validated_before_touching_drive(drive, store, bad):
    with pytest.raises(ValueError):
        store.load(bad, "c1", 3)
    with pytest.raises(ValueError):
        store.save("ana", bad, ClipAnnotation.empty(3))
    assert drive.files == {}
