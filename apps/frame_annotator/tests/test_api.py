"""HTTP API: auth, per-annotator isolation, per-frame labelling."""
import pytest

from frame_annotator.app import create_app
from frame_annotator.index_db import Clip, IndexDb, Sheet
from frame_annotator.store import LocalDirStore

TOKENS = {"tok-ana": "ana", "tok-bruno": "bruno"}


@pytest.fixture
def client(tmp_path):
    index = IndexDb(tmp_path / "index.sqlite")
    index.create()
    index.add_clip(Clip("c1", "E4", 10), [Sheet("c1", 0, "drive-file-1", 0, 10, 5, 160, 120)])
    app = create_app(
        index=index,
        store=LocalDirStore(tmp_path / "annotations"),
        tokens=TOKENS,
        image_source=lambda file_id: b"\xff\xd8jpeg-bytes-" + file_id.encode(),
    )
    return app.test_client()


def auth(name):
    return {"X-Annotator-Token": f"tok-{name}"}


def put(client, who, **body):
    return client.put("/api/clips/c1/labels", json=body, headers=auth(who))


def test_requests_without_a_valid_token_are_rejected(client):
    assert client.get("/api/clips").status_code == 401
    assert client.get("/api/clips", headers={"X-Annotator-Token": "nope"}).status_code == 401
    assert client.get("/sheet/c1/0").status_code == 401


def test_login_sets_an_http_only_cookie(client):
    response = client.post("/api/login", json={"token": "tok-ana"})

    assert response.status_code == 200
    assert response.get_json() == {"annotator": "ana"}
    assert "HttpOnly" in response.headers["Set-Cookie"]
    assert client.get("/api/me").get_json() == {"annotator": "ana"}


def test_login_rejects_bad_tokens(client):
    assert client.post("/api/login", json={"token": "wrong"}).status_code == 401


def test_a_fresh_clip_shows_no_existing_labels(client):
    data = client.get("/api/clips/c1", headers=auth("ana")).get_json()

    assert data["runs"] == []
    assert data["frame_count"] == 10
    assert data["labelled"] == 0
    assert data["sheets"][0]["cols"] == 5
    assert "drive_file_id" not in data["sheets"][0]  # links never reach the browser


def test_labelling_a_frame_range_persists_per_frame(client):
    response = put(client, "ana", start=2, end=4, obstacle="obstacle", deviation="left")

    assert response.status_code == 200
    body = response.get_json()
    assert body["runs"] == [[2, 4, "obstacle", "left", False]]
    assert body["labelled"] == 3
    assert body["complete"] is False
    again = client.get("/api/clips/c1", headers=auth("ana")).get_json()
    assert again["runs"] == body["runs"]


def test_unusable_flag_counts_as_done(client):
    body = put(client, "ana", start=0, end=9, unusable=True).get_json()

    assert body["complete"] is True


def test_annotators_cannot_see_each_others_work(client):
    put(client, "ana", start=0, end=9, obstacle="free_path")

    assert client.get("/api/clips/c1", headers=auth("bruno")).get_json()["runs"] == []
    listing = client.get("/api/clips", headers=auth("bruno")).get_json()
    assert listing["clips"][0]["labelled"] == 0
    assert client.get("/api/clips", headers=auth("ana")).get_json()["clips"][0]["labelled"] == 10


@pytest.mark.parametrize(
    "body",
    [
        {"start": 0, "end": 99, "obstacle": "obstacle"},
        {"start": 5, "end": 2, "obstacle": "obstacle"},
        {"start": 0, "end": 1, "obstacle": "wall"},
        {"start": 0, "end": 1, "obstacle": "free_path", "deviation": "left"},
        {"start": "0", "end": 1, "obstacle": "obstacle"},
        {"end": 1, "obstacle": "obstacle"},
    ],
)
def test_invalid_label_requests_are_400(client, body):
    assert put(client, "ana", **body).status_code == 400


def test_unknown_clip_is_404(client):
    response = client.put(
        "/api/clips/nope/labels", json={"start": 0, "end": 0, "obstacle": "free_path"},
        headers=auth("ana"),
    )

    assert response.status_code == 404


def test_sheet_images_are_proxied_for_authenticated_annotators(client):
    response = client.get("/sheet/c1/0", headers=auth("ana"))

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data == b"\xff\xd8jpeg-bytes-drive-file-1"
    assert client.get("/sheet/c1/7", headers=auth("ana")).status_code == 404


# ---- hardening (security review) ----


def test_non_json_bodies_cannot_write_labels(client):
    response = client.put(
        "/api/clips/c1/labels",
        data='{"start": 0, "end": 1, "obstacle": "free_path"}',
        content_type="text/plain",
        headers=auth("ana"),
    )

    assert response.status_code == 400
    assert client.get("/api/clips/c1", headers=auth("ana")).get_json()["labelled"] == 0


def test_repeated_bad_tokens_get_rate_limited(client):
    codes = [
        client.post("/api/login", json={"token": f"wrong-{i}"}).status_code for i in range(7)
    ]

    assert codes[:5] == [401] * 5
    assert codes[5:] == [429, 429]
    # even a valid token is refused while the address is locked out
    assert client.post("/api/login", json={"token": "tok-ana"}).status_code == 429


def test_logout_clears_the_session_cookie(client):
    client.post("/api/login", json={"token": "tok-ana"})
    assert client.post("/api/logout").status_code == 200

    assert client.get("/api/me").status_code == 401


def test_corrupt_stored_file_does_not_break_the_clip_list(client, tmp_path):
    folder = tmp_path / "annotations" / "ana"
    folder.mkdir(parents=True)
    (folder / "c1.json").write_text("{not json", encoding="utf-8")

    listing = client.get("/api/clips", headers=auth("ana"))

    assert listing.status_code == 200
    assert listing.get_json()["clips"][0]["labelled"] == 0
    opened = client.get("/api/clips/c1", headers=auth("ana"))
    assert opened.status_code == 409
    assert "c1.json" not in opened.get_data(as_text=True)


def test_error_bodies_do_not_leak_server_paths_or_drive_ids(client):
    text = client.get("/sheet/c1/9", headers=auth("ana")).get_data(as_text=True)

    assert "drive-file-1" not in text


def test_failing_image_source_returns_a_generic_502(tmp_path):
    index = IndexDb(tmp_path / "i.sqlite")
    index.create()
    index.add_clip(Clip("c1", "E4", 2), [Sheet("c1", 0, "secret-drive-id", 0, 2, 2, 8, 8)])

    def boom(file_id):
        raise RuntimeError(f"drive failed for {file_id}")

    app = create_app(index=index, store=LocalDirStore(tmp_path / "a"), tokens=TOKENS,
                     image_source=boom)
    response = app.test_client().get("/sheet/c1/0", headers=auth("ana"))

    assert response.status_code == 502
    assert "secret-drive-id" not in response.get_data(as_text=True)


def test_api_responses_are_not_cacheable_and_carry_nosniff(client):
    response = client.get("/api/clips", headers=auth("ana"))

    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_oversized_bodies_are_rejected(client):
    response = client.put(
        "/api/clips/c1/labels", json={"start": 0, "end": 1, "pad": "x" * 200_000},
        headers=auth("ana"),
    )

    assert response.status_code == 413


def test_clip_ids_with_odd_characters_are_404_not_500(client):
    for bad in ["..%2Fx", "a%20b", "%22%29"]:
        assert client.get(f"/api/clips/{bad}", headers=auth("ana")).status_code == 404
        assert client.get(f"/sheet/{bad}/0", headers=auth("ana")).status_code == 404


def test_only_the_index_page_is_served_unauthenticated(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/index.html").status_code == 404
