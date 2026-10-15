"""Server wiring from environment variables, end to end against a fake Drive."""
import json

import pytest

from frame_annotator.index_db import IndexDb
from frame_annotator.make_tokens import make_tokens
from frame_annotator.publish import jobs_from_manifest, publish_all, upload_index
from frame_annotator.wsgi import build_app

SHEETS = "sheets-folder"
ANNOTATIONS = "annotations-folder"
TOKEN = "t" * 40


@pytest.fixture
def published(tiny_dataset, tmp_path, drive):
    root, manifest = tiny_dataset
    index = IndexDb(tmp_path / "publisher.sqlite")
    index.create()
    publish_all(jobs_from_manifest(root, manifest), drive, SHEETS, index, tile=16, cols=2, rows=1)
    upload_index(drive, index.path, SHEETS)
    return drive


def _env(tmp_path, **extra):
    env = {
        "ANNOTATOR_TOKENS_JSON": json.dumps({TOKEN: "ana"}),
        "SHEETS_DRIVE_FOLDER_ID": SHEETS,
        "ANNOTATIONS_DRIVE_FOLDER_ID": ANNOTATIONS,
        "ANNOTATOR_INDEX_DB": str(tmp_path / "server" / "index.sqlite"),
        "IMAGE_CACHE_DIR": str(tmp_path / "server" / "cache"),
    }
    env.update(extra)
    return env


def test_server_boots_from_the_index_stored_on_drive(published, tmp_path):
    client = build_app(_env(tmp_path), drive=published).test_client()

    clips = client.get("/api/clips", headers={"X-Annotator-Token": TOKEN}).get_json()["clips"]

    assert [c["clip_id"] for c in clips] == ["E1_0207_01", "E1_0207_02"]


def test_labels_survive_a_server_restart(published, tmp_path):
    headers = {"X-Annotator-Token": TOKEN}
    first_app = build_app(_env(tmp_path), drive=published)
    response = first_app.test_client().put(
        "/api/clips/E1_0207_02/labels", headers=headers,
        json={"start": 0, "end": 1, "obstacle": "free_path"})
    assert response.status_code == 200
    assert first_app.extensions["annotation_store"].flush(10)  # write-behind

    second = build_app(_env(tmp_path), drive=published).test_client()
    opened = second.get("/api/clips/E1_0207_02", headers=headers).get_json()

    assert opened["complete"] is True


def test_sheet_images_are_fetched_through_drive(published, tmp_path):
    client = build_app(_env(tmp_path), drive=published).test_client()

    response = client.get("/sheet/E1_0207_01/0", headers={"X-Annotator-Token": TOKEN})

    assert response.status_code == 200
    assert response.data[:2] == b"\xff\xd8"


@pytest.mark.parametrize(
    "tokens",
    [json.dumps({"short": "ana"}), json.dumps({TOKEN: "bad id"}), "not json", json.dumps({})],
)
def test_weak_or_malformed_token_config_is_refused(published, tmp_path, tokens):
    with pytest.raises(ValueError):
        build_app(_env(tmp_path, ANNOTATOR_TOKENS_JSON=tokens), drive=published)


def test_make_tokens_creates_unique_long_tokens_for_safe_names():
    tokens = make_tokens(["ana", "bruno", "carla"])

    assert sorted(tokens.values()) == ["ana", "bruno", "carla"]
    assert len(set(tokens)) == 3
    assert all(len(token) >= 40 for token in tokens)
    with pytest.raises(ValueError):
        make_tokens(["ana", "ana"])
    with pytest.raises(ValueError):
        make_tokens(["../evil"])
