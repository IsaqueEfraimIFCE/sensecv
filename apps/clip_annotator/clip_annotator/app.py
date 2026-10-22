"""Flask API for the full-clip event annotator.

Authentication is a per-annotator secret token (header or HttpOnly cookie), as
in the frame annotator. Video refs (Drive file ids) stay on the server; the
browser only ever sees ``/video/<clip>`` URLs.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from flask import Flask, jsonify, request, send_file, send_from_directory

from .events import is_safe_id

log = logging.getLogger(__name__)

COOKIE_NAME = "clip_annotator_token"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30
MAX_BODY_BYTES = 64 * 1024
MAX_TOKEN_LEN = 256
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SEC = 60
VIDEO_MAX_AGE = 60 * 60 * 24
STATIC_DIR = Path(__file__).parent / "static"
CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
EDITABLE_FIELDS = ("events", "verdict", "done", "note")

# User-facing messages (the whole interface is in Portuguese).
UNREADABLE = "A anotação salva deste clipe está ilegível. Avise o responsável."
NOT_SIGNED_IN = "Sessão não iniciada. Abra o seu link pessoal ou informe o seu token."
UNKNOWN_CLIP = "Clipe não encontrado."
INVALID_ANNOTATION = "Anotação inválida."


class FailureLimiter:
    """Sliding-window counter of failed token attempts per client key.

    Successful logins deliberately do not reset it: otherwise an attacker who
    owns one valid token could interleave it with guesses forever."""

    def __init__(self, max_failures: int, window_sec: float, clock=time.monotonic):
        self._max = max_failures
        self._window = window_sec
        self._clock = clock
        self._failures: dict = defaultdict(list)
        self._lock = threading.Lock()

    def _recent(self, key) -> list:
        cutoff = self._clock() - self._window
        return [t for t in self._failures.get(key, ()) if t > cutoff]

    def blocked(self, key) -> bool:
        with self._lock:
            recent = self._recent(key)
            if recent:
                self._failures[key] = recent
            else:
                self._failures.pop(key, None)  # keep the table small
            return len(recent) >= self._max

    def record(self, key) -> None:
        with self._lock:
            self._failures[key] = self._recent(key) + [self._clock()]


class StorageUnavailable(Exception):
    """The annotation store (Drive) could not be reached; the client retries."""


def create_app(
    *,
    index,
    store,
    tokens: dict,
    video_source=None,
    secure_cookies: bool = False,
    client_ip_header: Optional[str] = None,
    max_failures: int = LOGIN_MAX_FAILURES,
    failure_window_sec: float = LOGIN_WINDOW_SEC,
) -> Flask:
    """``tokens`` maps secret token -> annotator id. ``video_source.path(ref)``
    returns a local file for a clip's ``video_ref``. Set ``secure_cookies=True``
    when TLS is terminated in front of the app (e.g. Fly.io), and
    ``client_ip_header`` to the header the proxy fills with the real client
    address (``Fly-Client-IP``): behind a proxy ``remote_addr`` is the proxy,
    and one person's wrong guesses would lock every annotator out."""
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.extensions["event_store"] = store  # lets the entry point flush on shutdown
    limiter = FailureLimiter(max_failures, failure_window_sec)
    known = [(hashlib.sha256(t.encode()).digest(), who) for t, who in tokens.items()]
    annotator_locks: dict = defaultdict(threading.Lock)
    locks_guard = threading.Lock()

    def lock_for(annotator_id: str) -> threading.Lock:
        with locks_guard:
            return annotator_locks[annotator_id]

    class Throttled(Exception):
        pass

    def match_token(candidate):
        """Annotator id for a token, None if wrong. Raises Throttled when the
        client address is locked out. Empty tokens are not counted as guesses."""
        if not isinstance(candidate, str) or not candidate:
            return None
        key = (client_ip_header and request.headers.get(client_ip_header)) or request.remote_addr
        if limiter.blocked(key):
            raise Throttled
        if len(candidate) > MAX_TOKEN_LEN:
            limiter.record(key)
            return None
        digest = hashlib.sha256(candidate.encode()).digest()
        found = None
        for stored, annotator_id in known:  # no early exit
            if hmac.compare_digest(stored, digest):
                found = annotator_id
        if found is None:
            limiter.record(key)
        return found

    def current_annotator():
        token = request.headers.get("X-Annotator-Token") or request.cookies.get(COOKIE_NAME)
        return match_token(token)

    def error(status: int, message: str, detail: str = ""):
        body = {"error": message}
        if detail:
            body["detail"] = detail
        return jsonify(body), status

    def find_clip(clip_id):
        return index.clip(clip_id) if is_safe_id(clip_id) else None

    def load_doc(who, clip):
        """Stored annotation, or None when the stored file is unusable.
        Raises StorageUnavailable when the store itself cannot be reached."""
        try:
            return store.load(who, clip)
        except (ValueError, KeyError, TypeError, AttributeError):
            log.exception("unreadable annotation for %s / %s", who, clip.clip_id)
            return None
        except Exception as exc:  # noqa: BLE001 - Drive/network errors are transient
            log.exception("annotation store failed for %s / %s", who, clip.clip_id)
            raise StorageUnavailable from exc

    def clip_payload(clip, doc):
        return {
            "clip_id": clip.clip_id,
            "location": clip.location,
            "recording": clip.recording,
            "duration_s": clip.duration_s,
            "fps": clip.fps,
            "frame_count": clip.frame_count,
            "video_url": f"/video/{quote(clip.clip_id, safe='')}",
            "annotation": doc.to_json(),
        }

    @app.errorhandler(Throttled)
    def throttled(_):
        return error(429, "Muitas tentativas incorretas. Tente novamente em 1 minuto.")

    @app.errorhandler(StorageUnavailable)
    def storage_unavailable(_):
        return error(503, "Armazenamento temporariamente indisponível. Tentando de novo…")

    @app.errorhandler(413)
    def too_large(_):
        return error(413, "Requisição grande demais.")

    @app.after_request
    def secure_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = CSP
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def home():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.post("/api/login")
    def login():
        body = request.get_json(silent=True)
        token = body.get("token") if isinstance(body, dict) else None
        who = match_token(token)
        if who is None:
            return error(401, "Token inválido.")
        response = jsonify({"annotator": who})
        response.set_cookie(
            COOKIE_NAME, token, httponly=True, samesite="Strict", path="/",
            secure=secure_cookies or request.is_secure, max_age=COOKIE_MAX_AGE,
        )
        return response

    @app.post("/api/logout")
    def logout():
        response = jsonify({"ok": True})
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.get("/api/health")
    def health():
        """Public. 503 when annotation writes to storage are stuck unsynced."""
        report = store.health() if hasattr(store, "health") else {"unsynced": 0, "ok": True}
        return jsonify(report), (200 if report["ok"] else 503)

    @app.get("/api/me")
    def me():
        who = current_annotator()
        return jsonify({"annotator": who}) if who else error(401, NOT_SIGNED_IN)

    @app.get("/api/clips")
    def list_clips():
        who = current_annotator()
        if who is None:
            return error(401, NOT_SIGNED_IN)
        try:
            summaries = store.summaries(who)
        except Exception as exc:  # noqa: BLE001 - Drive/network errors are transient
            log.exception("annotation store failed listing %s", who)
            raise StorageUnavailable from exc
        blank = {"status": "pendente", "events": 0, "verdict": None}
        return jsonify({"clips": [
            {"clip_id": c.clip_id, "location": c.location, "recording": c.recording,
             "duration_s": c.duration_s, **summaries.get(c.clip_id, blank)}
            for c in index.clips()
        ]})

    @app.get("/api/clips/<clip_id>")
    def get_clip(clip_id):
        who = current_annotator()
        if who is None:
            return error(401, NOT_SIGNED_IN)
        clip = find_clip(clip_id)
        if clip is None:
            return error(404, UNKNOWN_CLIP)
        doc = load_doc(who, clip)
        if doc is None:
            return error(409, UNREADABLE)
        return jsonify(clip_payload(clip, doc))

    @app.put("/api/clips/<clip_id>/annotation")
    def put_annotation(clip_id):
        who = current_annotator()
        if who is None:
            return error(401, NOT_SIGNED_IN)
        clip = find_clip(clip_id)
        if clip is None:
            return error(404, UNKNOWN_CLIP)
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) - set(EDITABLE_FIELDS):
            return error(400, INVALID_ANNOTATION, f"allowed fields: {', '.join(EDITABLE_FIELDS)}")
        with lock_for(who):
            current = load_doc(who, clip)
            if current is None:
                return error(409, UNREADABLE)
            try:
                doc = current.replace_with(body)
            except ValueError as exc:
                return error(400, INVALID_ANNOTATION, str(exc))
            try:
                store.save(who, clip.clip_id, doc)
            except Exception as exc:  # noqa: BLE001 - inline (non write-behind) stores
                log.exception("could not save %s / %s", who, clip.clip_id)
                raise StorageUnavailable from exc
        return jsonify(clip_payload(clip, doc))

    @app.get("/video/<clip_id>")
    def video(clip_id):
        if current_annotator() is None:
            return error(401, NOT_SIGNED_IN)
        clip = find_clip(clip_id)
        if clip is None:
            return error(404, "Vídeo não encontrado.")
        if video_source is None:
            return error(501, "Fonte de vídeos não configurada.")
        try:
            path = video_source.path(clip.video_ref)
        except Exception:  # noqa: BLE001 - never leak Drive details to the browser
            log.exception("video source failed for %s", clip_id)
            return error(502, "Vídeo temporariamente indisponível. Tente novamente.")
        response = send_file(path, mimetype="video/mp4", conditional=True, max_age=VIDEO_MAX_AGE)
        response.headers["Cache-Control"] = f"private, max-age={VIDEO_MAX_AGE}"
        return response

    return app
