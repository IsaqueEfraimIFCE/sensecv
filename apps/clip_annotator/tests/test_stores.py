import json

import pytest

from clip_annotator.drive_store import PROGRESS_NAME, DriveEventStore
from clip_annotator.events import ClipEvents
from clip_annotator.store import LocalDirStore, decode_doc, encode_doc
from conftest import desvio


def doc_for(clip, **payload):
    return ClipEvents.empty(clip.duration_s, clip.fps).replace_with(payload)


def test_encoded_file_is_self_describing(clip):
    data = json.loads(encode_doc(clip.clip_id, doc_for(clip, events=[desvio()])))
    assert data["clip_id"] == clip.clip_id
    assert (data["duration_s"], data["fps"]) == (10.0, 30.0)
    assert data["events"][0]["start_frame"] == 60
    assert data["updated_at"].endswith("Z")


def test_decode_refuses_another_clips_file(clip):
    raw = encode_doc("other", doc_for(clip))
    with pytest.raises(ValueError, match="belongs to"):
        decode_doc(raw, clip)


# -- local store -------------------------------------------------------------
def test_local_store_round_trip_and_isolation(tmp_path, clip):
    store = LocalDirStore(tmp_path)
    assert store.load("ana", clip).status == "pendente"
    store.save("ana", clip.clip_id, doc_for(clip, events=[desvio()], done=True))
    assert store.load("ana", clip).done
    assert store.load("bruno", clip).status == "pendente"
    assert store.summaries("ana") == {clip.clip_id: {"status": "concluido", "events": 1,
                                                      "verdict": "eventos"}}
    assert store.summaries("bruno") == {}


def test_local_store_skips_unreadable_files(tmp_path, clip):
    store = LocalDirStore(tmp_path)
    store.save("ana", clip.clip_id, doc_for(clip, verdict="sem_evento"))
    (tmp_path / "ana" / "broken.json").write_text("{", encoding="utf-8")
    assert list(store.summaries("ana")) == [clip.clip_id]


def test_local_store_rejects_unsafe_ids(tmp_path, clip):
    with pytest.raises(ValueError):
        LocalDirStore(tmp_path).save("../x", clip.clip_id, doc_for(clip))


# -- drive store -------------------------------------------------------------
def test_drive_store_writes_clip_file_and_progress(drive, clip):
    store = DriveEventStore(drive, "root")
    store.save("ana", clip.clip_id, doc_for(clip, events=[desvio()]))
    folder = drive.find("ana", "root")
    assert json.loads(drive.files[drive.find(f"{clip.clip_id}.json", folder)]["data"])["events"]
    progress = json.loads(drive.files[drive.find(PROGRESS_NAME, folder)]["data"])
    assert progress == {clip.clip_id: {"status": "andamento", "events": 1, "verdict": "eventos"}}


def test_drive_store_reads_back_after_restart(drive, clip):
    DriveEventStore(drive, "root").save("ana", clip.clip_id, doc_for(clip, verdict="inutilizavel"))
    fresh = DriveEventStore(drive, "root")
    assert fresh.load("ana", clip).verdict == "inutilizavel"
    assert fresh.summaries("ana")[clip.clip_id]["verdict"] == "inutilizavel"
    assert fresh.load("bruno", clip).status == "pendente"
    assert fresh.summaries("bruno") == {}


def test_drive_store_rebuilds_a_broken_progress_file(drive, clip):
    DriveEventStore(drive, "root").save("ana", clip.clip_id, doc_for(clip, events=[desvio()]))
    folder = drive.find("ana", "root")
    drive.files[drive.find(PROGRESS_NAME, folder)]["data"] = b"[1, 2]"
    assert DriveEventStore(drive, "root").summaries("ana")[clip.clip_id]["events"] == 1


def test_write_behind_returns_before_drive_and_flushes(drive, clip):
    store = DriveEventStore(drive, "root", write_behind=True, coalesce_sec=0.01)
    store.save("ana", clip.clip_id, doc_for(clip, events=[desvio()]))
    assert store.load("ana", clip).events  # served from memory right away
    assert store.summaries("ana")[clip.clip_id]["events"] == 1
    assert store.flush(5)
    assert store.pending_writes() == 0
    assert DriveEventStore(drive, "root").load("ana", clip).events
    assert store.close(5)


def test_write_behind_retries_failed_uploads(drive, clip):
    drive.fail_uploads = 2
    store = DriveEventStore(drive, "root", write_behind=True, coalesce_sec=0.01, retry_sec=0.01)
    store.save("ana", clip.clip_id, doc_for(clip, verdict="sem_evento"))
    assert store.flush(5)
    assert DriveEventStore(drive, "root").load("ana", clip).verdict == "sem_evento"
    assert store.health() == {"unsynced": 0, "ok": True}
    store.close(5)


def test_decode_rejects_non_object_json(clip):
    with pytest.raises(ValueError, match="object"):
        decode_doc(b"[1, 2]", clip)


def test_slow_download_does_not_replace_a_newer_save(drive, clip):
    DriveEventStore(drive, "root").save("ana", clip.clip_id, doc_for(clip, note="antiga"))
    store = DriveEventStore(drive, "root")
    real_download = drive.download
    fired = []

    def download_while_a_save_lands(file_id):
        data = real_download(file_id)
        if not fired:
            fired.append(True)
            store.save("ana", clip.clip_id, doc_for(clip, note="nova"))  # e.g. another tab
        return data

    drive.download = download_while_a_save_lands
    assert store.load("ana", clip).note == "nova"
    drive.download = real_download
    assert store.load("ana", clip).note == "nova"


def test_concurrent_progress_reads_share_one_state(drive, clip):
    DriveEventStore(drive, "root").save("ana", clip.clip_id, doc_for(clip, events=[desvio()]))
    store = DriveEventStore(drive, "root")
    folder = drive.find("ana", "root")
    real_find = drive.find
    seen, fired = [], []

    def find_racing_another_reader(name, parent):
        if name == PROGRESS_NAME and not fired:
            fired.append(True)
            seen.append(store._progress_state("ana", folder))  # the other reader wins first
        return real_find(name, parent)

    drive.find = find_racing_another_reader
    assert store._progress_state("ana", folder) is seen[0]


def test_write_behind_keeps_only_the_newest_version(drive, clip):
    store = DriveEventStore(drive, "root", write_behind=True, coalesce_sec=0.2)
    store.save("ana", clip.clip_id, doc_for(clip, note="primeira"))
    store.save("ana", clip.clip_id, doc_for(clip, note="segunda"))
    assert store.close(5)
    assert DriveEventStore(drive, "root").load("ana", clip).note == "segunda"
