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


def _gate_uploads(drive, gate):
    """Make every upload wait for ``gate`` (a threading.Event); returns a call log."""
    calls = []
    real = drive.upload

    def gated(name, data, parent_id, mime="application/octet-stream", file_id=None):
        gate.wait(10)
        calls.append(name)
        return real(name, data, parent_id, mime, file_id=file_id)

    drive.upload = gated
    return calls


def _write_behind(drive, **kwargs):
    return DriveAnnotationStore(drive, ROOT, write_behind=True, coalesce_sec=0.01,
                                retry_sec=0.01, **kwargs)


def test_write_behind_save_returns_without_waiting_for_drive(drive):
    import threading
    import time

    gate = threading.Event()
    _gate_uploads(drive, gate)
    store = _write_behind(drive)

    started = time.perf_counter()
    store.save("ana", "c1", _clip())
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert store.load("ana", "c1", 8) == _clip()  # visible immediately
    assert store.pending_writes() == 1
    gate.set()
    assert store.flush(5)
    assert store.pending_writes() == 0
    assert DriveAnnotationStore(drive, ROOT).load("ana", "c1", 8) == _clip()
    store.close()


def test_rapid_saves_are_coalesced_and_the_last_one_wins(drive):
    import threading

    gate = threading.Event()
    calls = _gate_uploads(drive, gate)
    store = _write_behind(drive)
    final = None
    for upto in range(5):
        final = _clip(upto=upto)
        store.save("ana", "c1", final)

    gate.set()
    assert store.flush(5)

    assert calls.count("c1.json") <= 2
    assert DriveAnnotationStore(drive, ROOT).load("ana", "c1", 8) == final
    store.close()


def test_failed_writes_are_retried_until_they_succeed(drive):
    real = drive.upload
    failures = {"left": 3}

    def flaky(name, data, parent_id, mime="application/octet-stream", file_id=None):
        if failures["left"] > 0:
            failures["left"] -= 1
            raise RuntimeError("drive is down")
        return real(name, data, parent_id, mime, file_id=file_id)

    drive.upload = flaky
    store = _write_behind(drive)
    store.save("ana", "c1", _clip())

    assert store.flush(5)
    assert DriveAnnotationStore(drive, ROOT).load("ana", "c1", 8) == _clip()
    store.close()


def test_health_reports_unsynced_writes(drive):
    import threading

    gate = threading.Event()
    _gate_uploads(drive, gate)
    store = _write_behind(drive)

    assert store.health() == {"unsynced": 0, "ok": True}
    store.save("ana", "c1", _clip())
    assert store.health()["unsynced"] == 1
    gate.set()
    store.flush(5)
    assert store.health() == {"unsynced": 0, "ok": True}
    store.close()


def test_close_flushes_pending_work(drive):
    store = _write_behind(drive)
    store.save("ana", "c1", _clip())
    store.close()

    assert DriveAnnotationStore(drive, ROOT).load("ana", "c1", 8) == _clip()


def test_progress_lists_unsaved_work_immediately(drive):
    import threading

    gate = threading.Event()
    _gate_uploads(drive, gate)
    store = _write_behind(drive)
    store.save("ana", "c1", ClipAnnotation.empty(4).apply(0, 1, obstacle="free_path"))

    assert store.labelled_counts("ana") == {"c1": 2}
    gate.set()
    store.close()


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a b"])
def test_ids_are_validated_before_touching_drive(drive, store, bad):
    with pytest.raises(ValueError):
        store.load(bad, "c1", 3)
    with pytest.raises(ValueError):
        store.save("ana", bad, ClipAnnotation.empty(3))
    assert drive.files == {}
