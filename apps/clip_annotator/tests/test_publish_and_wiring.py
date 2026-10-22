import json
import os
import threading

import pytest

from clip_annotator.dev import make_dev_app
from clip_annotator.index_db import Clip, IndexDb
from clip_annotator.make_tokens import main as make_tokens_main
from clip_annotator.make_tokens import make_tokens
from clip_annotator.publish import INDEX_NAME, discover_clips, publish_all, upload_index
from clip_annotator.videos import CachedVideoSource, LocalVideoSource
from clip_annotator.wsgi import build_app, parse_tokens
from clip_annotator.mp4 import mp4_duration
from conftest import FAKE_MP4, mp4_bytes, write_recording


# -- discovery ---------------------------------------------------------------
def test_discovery_keeps_full_recordings_only(dataset):
    clips = discover_clips(dataset)
    assert [c.clip_id for c in clips] == [
        "SenseCV-01-01-2026-EXEMPLO__07", "SenseCV-01-01-2026-EXEMPLO__08",
        "SenseCV-02-01-2026-OUTRO__01"]
    first = clips[0]
    assert (first.location, first.recording, first.duration_s, first.fps, first.frame_count) == (
        "SenseCV-01-01-2026-EXEMPLO", "07", 1.2, 30.0, 30)


def test_duration_comes_from_the_mp4_header_not_the_metadata(tmp_path):
    folder = tmp_path / "root" / "col" / "01"
    write_recording(folder, duration=8.6)
    (folder / "metadata.json").write_text('{"duration_seconds": 9.259, "fps_selected": 30}',
                                          encoding="utf-8")
    assert discover_clips(tmp_path / "root")[0].duration_s == 8.6


def test_discovery_falls_back_to_frame_timestamps(tmp_path):
    folder = tmp_path / "root" / "col" / "01"
    write_recording(folder, frames=31, fps=30, header=False)
    (folder / "metadata.json").write_text("{}", encoding="utf-8")
    clip = discover_clips(tmp_path / "root")[0]
    assert clip.fps == pytest.approx(30.0, abs=0.05)
    assert clip.duration_s == pytest.approx(1.0 + 1 / 30, abs=0.01)


def test_metadata_duration_is_the_last_resort(tmp_path):
    folder = tmp_path / "root" / "col" / "01"
    write_recording(folder, frames=1, duration=4.0, header=False)
    assert discover_clips(tmp_path / "root")[0].duration_s == 4.0


@pytest.mark.parametrize("data, expected", [
    (mp4_bytes(8.601), 8.601),
    (mp4_bytes(216.5, version=1, timescale=90000), 216.5),
    (b"\x00\x00\x00\x08free", None),
    (b"\x00\x00\x00\xffmoov\x00", None),   # box claims more bytes than the file has
    (b"", None),
])
def test_mp4_duration(tmp_path, data, expected):
    path = tmp_path / "v.mp4"
    path.write_bytes(data)
    assert mp4_duration(path) == (pytest.approx(expected) if expected else None)


def test_discovery_sanitises_ids(tmp_path):
    write_recording(tmp_path / "root" / "Coleta Nova" / "gravação 1")
    assert discover_clips(tmp_path / "root")[0].clip_id == "Coleta_Nova__grava_o_1"


def test_discovery_skips_broken_recordings(tmp_path):
    folder = tmp_path / "root" / "col" / "bad"
    write_recording(folder)
    (folder / "frames.json").write_text('{"frames": []}', encoding="utf-8")
    assert discover_clips(tmp_path / "root") == []


# -- publish -----------------------------------------------------------------
def test_publish_uploads_videos_and_is_resumable(tmp_path, dataset, drive):
    index = IndexDb(tmp_path / INDEX_NAME)
    index.create()
    sources = discover_clips(dataset)
    summary = publish_all(sources, drive, "videos", index, workers=2)
    assert (summary.published, summary.skipped, summary.failed) == (3, 0, [])
    clip = index.clip("SenseCV-01-01-2026-EXEMPLO__07")
    assert drive.files[clip.video_ref] == {"name": "SenseCV-01-01-2026-EXEMPLO__07.mp4",
                                           "parent": "videos", "data": FAKE_MP4, "mime": "video/mp4"}
    assert clip.video_bytes == len(FAKE_MP4)
    again = publish_all(sources, drive, "videos", index)
    assert (again.published, again.skipped) == (0, 3)
    upload_index(drive, index.path, "videos")
    assert drive.find(INDEX_NAME, "videos")


def test_publish_reports_failed_clips(tmp_path, dataset, drive):
    index = IndexDb(tmp_path / INDEX_NAME)
    index.create()
    drive.fail_uploads = 1
    summary = publish_all(discover_clips(dataset), drive, "videos", index, workers=1)
    assert summary.published == 2
    assert len(summary.failed) == 1


def test_index_rejects_bad_rows(tmp_path):
    index = IndexDb(tmp_path / INDEX_NAME)
    index.create()
    row = Clip("ok", "loc", "01", 1.0, 30.0, 30, "ref", 10)
    index.add_clip(row)
    with pytest.raises(ValueError):
        index.add_clip(row)
    with pytest.raises(ValueError):
        index.add_clip(Clip("../x", "loc", "01", 1.0, 30.0, 30, "ref", 10))
    with pytest.raises(ValueError):
        index.add_clip(Clip("long", "loc", "01", 99_999.0, 30.0, 30, "ref", 10))


# -- video sources -----------------------------------------------------------
def test_cached_video_source_downloads_once_under_concurrency(tmp_path, drive):
    file_id = drive.upload("v.mp4", FAKE_MP4, "videos")
    source = CachedVideoSource(drive, tmp_path / "cache")
    paths = []
    threads = [threading.Thread(target=lambda: paths.append(source.path(file_id))) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert drive.downloads == 1
    assert {p.read_bytes() for p in paths} == {FAKE_MP4}


def test_cached_video_source_evicts_least_recent(tmp_path, drive):
    ids = [drive.upload(f"{i}.mp4", FAKE_MP4, "videos") for i in range(3)]
    source = CachedVideoSource(drive, tmp_path / "cache", max_bytes=int(len(FAKE_MP4) * 2.5))
    first = source.path(ids[0])
    os.utime(first, (1, 1))
    source.path(ids[1])
    source.path(ids[2])
    assert not first.exists()
    assert len(list((tmp_path / "cache").glob("*.mp4"))) == 2


def test_cached_video_source_rejects_odd_ids(tmp_path, drive):
    with pytest.raises(ValueError):
        CachedVideoSource(drive, tmp_path).path("../../etc/passwd")


def test_local_video_source_stays_inside_the_root(dataset):
    source = LocalVideoSource(dataset)
    assert source.path("SenseCV-01-01-2026-EXEMPLO/07/video.mp4").read_bytes() == FAKE_MP4
    for ref in ("../outside.mp4", "SenseCV-01-01-2026-EXEMPLO/07/missing.mp4"):
        with pytest.raises(ValueError):
            source.path(ref)


# -- wiring ------------------------------------------------------------------
@pytest.mark.parametrize("raw", [None, "[]", "{}", '{"short": "ana"}', json.dumps({"x" * 40: "a b"})])
def test_parse_tokens_rejects_bad_config(raw):
    with pytest.raises(ValueError):
        parse_tokens(raw)


def test_build_app_downloads_the_index_and_serves(tmp_path, dataset, drive):
    index = IndexDb(tmp_path / "published.sqlite")
    index.create()
    publish_all(discover_clips(dataset), drive, "videos", index)
    upload_index(drive, index.path, "videos")
    token = "t" * 40
    env = {"ANNOTATOR_TOKENS_JSON": json.dumps({token: "ana"}), "VIDEOS_DRIVE_FOLDER_ID": "videos",
           "ANNOTATIONS_DRIVE_FOLDER_ID": "ann", "ANNOTATOR_INDEX_DB": str(tmp_path / "srv" / "i.sqlite"),
           "VIDEO_CACHE_DIR": str(tmp_path / "cache")}
    app = build_app(env, drive=drive)
    client = app.test_client()
    clips = client.get("/api/clips", headers={"X-Annotator-Token": token}).get_json()["clips"]
    assert len(clips) == 3
    video = client.get(f"/video/{clips[0]['clip_id']}", headers={"X-Annotator-Token": token})
    assert video.data == FAKE_MP4
    assert app.extensions["event_store"].close(5)


def test_build_app_requires_a_published_index(tmp_path, drive):
    env = {"ANNOTATOR_TOKENS_JSON": json.dumps({"t" * 40: "ana"}), "VIDEOS_DRIVE_FOLDER_ID": "videos",
           "ANNOTATIONS_DRIVE_FOLDER_ID": "ann", "ANNOTATOR_INDEX_DB": str(tmp_path / "i.sqlite"),
           "VIDEO_CACHE_DIR": str(tmp_path / "cache")}
    with pytest.raises(RuntimeError, match="publisher"):
        build_app(env, drive=drive)
    with pytest.raises(ValueError, match="VIDEOS_DRIVE_FOLDER_ID"):
        build_app({**env, "VIDEOS_DRIVE_FOLDER_ID": ""}, drive=drive)


def test_make_tokens_writes_a_new_file_only(tmp_path, capsys):
    out = tmp_path / "annotators.json"
    assert make_tokens_main(["ana", "bruno", "--out", str(out)]) == 0
    tokens = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(tokens.values()) == ["ana", "bruno"]
    assert all(len(t) >= 40 for t in tokens)
    assert "#token=" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        make_tokens_main(["carla", "--out", str(out)])
    with pytest.raises(ValueError):
        make_tokens(["ana", "ana"])
    with pytest.raises(ValueError):
        make_tokens(["../evil"])


def test_dev_app_reuses_tokens_between_runs(tmp_path, dataset):
    _, first, index = make_dev_app(dataset, tmp_path / "dev", ["ana"])
    _, second, _ = make_dev_app(dataset, tmp_path / "dev", ["ana", "bruno"])
    assert set(first) <= set(second)
    assert sorted(second.values()) == ["ana", "bruno"]
    assert len(index.clips()) == 3
