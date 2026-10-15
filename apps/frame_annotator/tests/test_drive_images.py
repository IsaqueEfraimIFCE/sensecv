"""Sheet images come from Drive once, then from a size-capped disk cache."""
import pytest

from frame_annotator.drive_images import CachedImageSource


def _drive_with(drive, **named):
    return {name: drive.upload(name, data, "sheets") for name, data in named.items()}


def test_second_read_is_served_from_disk(drive, tmp_path):
    ids = _drive_with(drive, a=b"A" * 100)
    source = CachedImageSource(drive, tmp_path / "cache", max_bytes=10_000)

    assert source(ids["a"]) == b"A" * 100
    downloads = drive.downloads
    assert source(ids["a"]) == b"A" * 100
    assert drive.downloads == downloads


def test_oldest_entries_are_evicted_when_over_the_cap(drive, tmp_path):
    ids = _drive_with(drive, a=b"A" * 400, b=b"B" * 400, c=b"C" * 400)
    source = CachedImageSource(drive, tmp_path / "cache", max_bytes=1000)

    for name in ("a", "b", "c"):
        source(ids[name])
    downloads = drive.downloads
    source(ids["c"])  # still cached
    assert drive.downloads == downloads
    source(ids["a"])  # evicted, fetched again
    assert drive.downloads == downloads + 1


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "a b", "x" * 300])
def test_unsafe_file_ids_never_reach_the_filesystem(drive, tmp_path, bad):
    source = CachedImageSource(drive, tmp_path / "cache")

    with pytest.raises(ValueError):
        source(bad)
