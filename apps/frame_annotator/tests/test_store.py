"""Each annotator's work is stored separately and can be resumed."""
import json

import pytest

from frame_annotator.labels import ClipAnnotation
from frame_annotator.store import LocalDirStore


@pytest.fixture
def store(tmp_path):
    return LocalDirStore(tmp_path / "annotations")


def test_missing_clip_loads_as_untouched(store):
    clip = store.load("ana", "c1", 8)

    assert clip == ClipAnnotation.empty(8)


def test_save_then_load_resumes_exactly(store):
    clip = ClipAnnotation.empty(8).apply(0, 3, obstacle="obstacle", deviation="left")
    store.save("ana", "c1", clip)

    assert store.load("ana", "c1", 8) == clip


def test_annotators_are_isolated(store):
    store.save("ana", "c1", ClipAnnotation.empty(4).apply(0, 3, obstacle="free_path"))

    assert store.load("bruno", "c1", 4) == ClipAnnotation.empty(4)


def test_labelled_counts_cover_only_that_annotator(store):
    store.save("ana", "c1", ClipAnnotation.empty(4).apply(0, 1, obstacle="free_path"))
    store.save("ana", "c2", ClipAnnotation.empty(3).apply(0, 2, unusable=True))
    store.save("bruno", "c1", ClipAnnotation.empty(4).apply(0, 3, obstacle="free_path"))

    assert store.labelled_counts("ana") == {"c1": 2, "c2": 3}
    assert store.labelled_counts("nobody") == {}


def test_save_is_atomic_json_with_runs_and_timestamp(store, tmp_path):
    store.save("ana", "c1", ClipAnnotation.empty(2).apply(0, 1, obstacle="free_path"))

    data = json.loads((tmp_path / "annotations" / "ana" / "c1.json").read_text("utf-8"))

    assert data["runs"] == [[0, 1, "free_path", "none", False]]
    assert data["frame_count"] == 2
    assert data["updated_at"].endswith("Z")
    assert not list((tmp_path / "annotations" / "ana").glob("*.tmp"))


def test_frame_count_mismatch_is_an_error_not_silent_data_loss(store):
    store.save("ana", "c1", ClipAnnotation.empty(4).apply(0, 3, obstacle="free_path"))

    with pytest.raises(ValueError):
        store.load("ana", "c1", 9)


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a b", "x" * 200])
def test_ids_cannot_escape_the_store_root(store, bad):
    with pytest.raises(ValueError):
        store.load(bad, "c1", 3)
    with pytest.raises(ValueError):
        store.load("ana", bad, 3)
