import pytest

from clip_annotator.app import COOKIE_NAME, FailureLimiter, create_app
from clip_annotator.index_db import IndexDb
from clip_annotator.publish import build_local_index
from clip_annotator.store import LocalDirStore
from clip_annotator.videos import LocalVideoSource
from conftest import FAKE_MP4, desvio

ANA = "a" * 40
BRUNO = "b" * 40
CLIP = "SenseCV-01-01-2026-EXEMPLO__07"


@pytest.fixture
def app(tmp_path, dataset):
    index = IndexDb(tmp_path / "index.sqlite")
    build_local_index(dataset, index)
    return create_app(index=index, store=LocalDirStore(tmp_path / "ann"),
                      tokens={ANA: "ana", BRUNO: "bruno"}, video_source=LocalVideoSource(dataset))


@pytest.fixture
def client(app):
    return app.test_client()


def as_(token):
    return {"X-Annotator-Token": token}


def put(client, body, token=ANA, clip=CLIP):
    return client.put(f"/api/clips/{clip}/annotation", json=body, headers=as_(token))


def test_page_and_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Anotador de clipes" in response.data
    assert "media-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["X-Frame-Options"] == "DENY"


@pytest.mark.parametrize("path", ["/api/me", "/api/clips", f"/api/clips/{CLIP}", f"/video/{CLIP}"])
def test_everything_needs_a_token(client, path):
    assert client.get(path).status_code == 401


def test_login_sets_an_httponly_cookie(client):
    response = client.post("/api/login", json={"token": ANA})
    assert response.get_json() == {"annotator": "ana"}
    cookie = response.headers["Set-Cookie"]
    assert COOKIE_NAME in cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert client.get("/api/me").get_json() == {"annotator": "ana"}
    client.post("/api/logout")
    assert client.get("/api/me").status_code == 401


def test_wrong_tokens_are_throttled(client):
    for _ in range(5):
        assert client.post("/api/login", json={"token": "x" * 40}).status_code == 401
    assert client.post("/api/login", json={"token": ANA}).status_code == 429


def test_list_shows_full_clips_only(client):
    clips = client.get("/api/clips", headers=as_(ANA)).get_json()["clips"]
    assert [c["clip_id"] for c in clips] == [
        CLIP, "SenseCV-01-01-2026-EXEMPLO__08", "SenseCV-02-01-2026-OUTRO__01"]
    assert clips[0] == {"clip_id": CLIP, "location": "SenseCV-01-01-2026-EXEMPLO",
                        "recording": "07", "duration_s": 1.2, "status": "pendente",
                        "events": 0, "verdict": None}


def test_get_clip_hides_the_video_ref(client):
    data = client.get(f"/api/clips/{CLIP}", headers=as_(ANA)).get_json()
    assert data["video_url"] == f"/video/{CLIP}"
    assert (data["fps"], data["frame_count"]) == (30.0, 30)
    assert "video_ref" not in data
    assert data["annotation"]["status"] == "pendente"


def test_unknown_or_unsafe_clip(client):
    assert client.get("/api/clips/nope", headers=as_(ANA)).status_code == 404
    assert client.get("/api/clips/..%2Fx", headers=as_(ANA)).status_code == 404
    assert put(client, {}, clip="nope").status_code == 404


def test_save_events_and_conclude(client):
    reply = put(client, {"events": [desvio(0.2, 0.9, "direita")], "note": "ok"})
    assert reply.status_code == 200
    event = reply.get_json()["annotation"]["events"][0]
    assert (event["side"], event["start_frame"], event["end_frame"]) == ("direita", 6, 27)
    reply = put(client, {"events": [desvio(0.2, 0.9, "direita")], "done": True})
    assert reply.get_json()["annotation"]["status"] == "concluido"
    row = client.get("/api/clips", headers=as_(ANA)).get_json()["clips"][0]
    assert (row["status"], row["events"], row["verdict"]) == ("concluido", 1, "eventos")


def test_annotators_are_isolated(client):
    put(client, {"verdict": "sem_evento", "done": True})
    row = client.get("/api/clips", headers=as_(BRUNO)).get_json()["clips"][0]
    assert row["status"] == "pendente"
    assert client.get(f"/api/clips/{CLIP}", headers=as_(BRUNO)).get_json()["annotation"]["verdict"] is None


@pytest.mark.parametrize("body", [
    {"events": [desvio(0.2, 5.0)]},          # beyond the 1.2 s clip (+0.5 s slack)
    {"done": True},                           # nothing to conclude
    {"events": [], "owner": "bruno"},         # unknown field
    [1, 2],
])
def test_invalid_annotations_are_rejected_and_not_saved(client, body):
    reply = put(client, body)
    assert reply.status_code == 400
    assert reply.get_json()["error"] == "Anotação inválida."
    data = client.get(f"/api/clips/{CLIP}", headers=as_(ANA)).get_json()
    assert data["annotation"]["status"] == "pendente"


def test_body_size_is_capped(client):
    reply = client.put(f"/api/clips/{CLIP}/annotation", data=b"x" * 70_000,
                       headers={**as_(ANA), "Content-Type": "application/json"})
    assert reply.status_code == 413


def test_unreadable_stored_file_is_reported(client, tmp_path):
    folder = tmp_path / "ann" / "ana"
    folder.mkdir(parents=True)
    (folder / f"{CLIP}.json").write_text("{", encoding="utf-8")
    assert client.get(f"/api/clips/{CLIP}", headers=as_(ANA)).status_code == 409
    assert put(client, {"note": "x"}).status_code == 409


def test_video_is_streamed_with_range_support(client):
    full = client.get(f"/video/{CLIP}", headers=as_(ANA))
    assert full.status_code == 200
    assert full.mimetype == "video/mp4"
    assert full.data == FAKE_MP4
    assert full.headers["Cache-Control"] == "private, max-age=86400"
    part = client.get(f"/video/{CLIP}", headers={**as_(ANA), "Range": "bytes=10-19"})
    assert part.status_code == 206
    assert part.data == FAKE_MP4[10:20]
    assert part.headers["Content-Range"] == f"bytes 10-19/{len(FAKE_MP4)}"


def test_video_source_failures_do_not_leak(tmp_path, dataset):
    class Broken:
        def path(self, ref):
            raise RuntimeError("drive said: secret file id fid123")

    index = IndexDb(tmp_path / "index.sqlite")
    build_local_index(dataset, index)
    client = create_app(index=index, store=LocalDirStore(tmp_path / "ann"), tokens={ANA: "ana"},
                        video_source=Broken()).test_client()
    reply = client.get(f"/video/{CLIP}", headers=as_(ANA))
    assert reply.status_code == 502
    assert b"fid123" not in reply.data


def test_throttle_is_per_client_behind_a_proxy(tmp_path, dataset):
    index = IndexDb(tmp_path / "index.sqlite")
    build_local_index(dataset, index)
    client = create_app(index=index, store=LocalDirStore(tmp_path / "ann"), tokens={ANA: "ana"},
                        client_ip_header="Fly-Client-IP").test_client()
    attacker = {"Fly-Client-IP": "203.0.113.9"}
    for _ in range(5):
        client.post("/api/login", json={"token": "x" * 40}, headers=attacker)
    assert client.post("/api/login", json={"token": ANA}, headers=attacker).status_code == 429
    honest = {**as_(ANA), "Fly-Client-IP": "198.51.100.7"}
    assert client.get("/api/clips", headers=honest).status_code == 200


def test_storage_outage_is_503_not_unreadable(tmp_path, dataset):
    class Down(LocalDirStore):
        def load(self, annotator_id, clip):
            raise ConnectionError("drive unreachable")

        def summaries(self, annotator_id):
            raise TimeoutError("drive unreachable")

    index = IndexDb(tmp_path / "index.sqlite")
    build_local_index(dataset, index)
    client = create_app(index=index, store=Down(tmp_path / "ann"), tokens={ANA: "ana"}).test_client()
    assert client.get(f"/api/clips/{CLIP}", headers=as_(ANA)).status_code == 503
    assert put(client, {"note": "x"}).status_code == 503
    assert client.get("/api/clips", headers=as_(ANA)).status_code == 503


def test_non_object_stored_file_is_unreadable_not_500(client, tmp_path):
    folder = tmp_path / "ann" / "ana"
    folder.mkdir(parents=True)
    (folder / f"{CLIP}.json").write_text("[1, 2]", encoding="utf-8")
    assert client.get(f"/api/clips/{CLIP}", headers=as_(ANA)).status_code == 409


def test_failure_limiter_forgets_quiet_clients():
    now = [0.0]
    limiter = FailureLimiter(2, 10, clock=lambda: now[0])
    limiter.record("a")
    now[0] = 11
    assert not limiter.blocked("a")
    assert limiter._failures == {}


def test_health_is_public(client):
    assert client.get("/api/health").get_json() == {"unsynced": 0, "ok": True}
