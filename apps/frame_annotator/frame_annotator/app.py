"""Flask API for the per-frame annotator.

Authentication is a per-annotator secret token (header or HttpOnly cookie).
Drive file ids stay on the server; the browser only ever sees
``/sheet/<clip>/<n>`` URLs.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, jsonify, request, send_from_directory

from .labels import is_safe_id

log = logging.getLogger(__name__)

LABEL_FIELDS = ("obstacle", "deviation", "unusable")
COOKIE_NAME = "annotator_token"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30
MAX_BODY_BYTES = 64 * 1024
MAX_TOKEN_LEN = 256
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SEC = 60
STATIC_DIR = Path(__file__).parent / "static"
CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
UNREADABLE = "stored annotation is unreadable; please contact the administrator"


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
        recent = [t for t in self._failures[key] if t > cutoff]
        self._failures[key] = recent
        return recent

    def blocked(self, key) -> bool:
        with self._lock:
            return len(self._recent(key)) >= self._max

    def record(self, key) -> None:
        with self._lock:
            self._recent(key).append(self._clock())


def create_app(
    *,
    index,
    store,
    tokens: dict,
    image_source=None,
    secure_cookies: bool = False,
    max_failures: int = LOGIN_MAX_FAILURES,
    failure_window_sec: float = LOGIN_WINDOW_SEC,
) -> Flask:
    """``tokens`` maps secret token -> annotator id. ``image_source`` maps a
    Drive file id to JPEG bytes. Set ``secure_cookies=True`` when TLS is
    terminated in front of the app (e.g. Fly.io)."""
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
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
        key = request.remote_addr
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

    def error(status: int, message: str):
        return jsonify({"error": message}), status

    @app.errorhandler(Throttled)
    def throttled(_):
        return error(429, "too many failed attempts; try again in a minute")

    @app.errorhandler(413)
    def too_large(_):
        return error(413, "request body too large")

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
            return error(401, "invalid token")
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

    @app.get("/api/me")
    def me():
        who = current_annotator()
        return jsonify({"annotator": who}) if who else error(401, "not signed in")

    @app.get("/api/clips")
    def list_clips():
        who = current_annotator()
        if who is None:
            return error(401, "not signed in")
        counts = store.labelled_counts(who)
        return jsonify({"clips": [
            {"clip_id": c.clip_id, "location": c.location, "frame_count": c.frame_count,
             "labelled": counts.get(c.clip_id, 0)}
            for c in index.clips()
        ]})

    def clip_payload(clip, annotation):
        return {
            "clip_id": clip.clip_id,
            "location": clip.location,
            "frame_count": clip.frame_count,
            "runs": annotation.to_runs(),
            "labelled": annotation.labelled_count,
            "complete": annotation.is_complete,
        }

    def load_annotation(who, clip):
        """Stored annotation, or None when the stored file is unusable."""
        try:
            return store.load(who, clip.clip_id, clip.frame_count)
        except (ValueError, KeyError, TypeError, OSError):
            log.exception("unreadable annotation for %s / %s", who, clip.clip_id)
            return None

    @app.get("/api/clips/<clip_id>")
    def get_clip(clip_id):
        who = current_annotator()
        if who is None:
            return error(401, "not signed in")
        clip = index.clip(clip_id) if is_safe_id(clip_id) else None
        if clip is None:
            return error(404, "unknown clip")
        annotation = load_annotation(who, clip)
        if annotation is None:
            return error(409, UNREADABLE)
        payload = clip_payload(clip, annotation)
        safe = quote(clip_id, safe="")
        payload["sheets"] = [
            {"sheet_index": s.sheet_index, "first_frame": s.first_frame,
             "frame_count": s.frame_count, "cols": s.cols, "tile_w": s.tile_w,
             "tile_h": s.tile_h, "url": f"/sheet/{safe}/{s.sheet_index}"}
            for s in index.sheets(clip_id)
        ]
        return jsonify(payload)

    @app.put("/api/clips/<clip_id>/labels")
    def put_labels(clip_id):
        who = current_annotator()
        if who is None:
            return error(401, "not signed in")
        clip = index.clip(clip_id) if is_safe_id(clip_id) else None
        if clip is None:
            return error(404, "unknown clip")
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not _is_int(body.get("start")) or not _is_int(body.get("end")):
            return error(400, "start and end must be integers")
        fields = {name: body[name] for name in LABEL_FIELDS if name in body}
        with lock_for(who):
            current = load_annotation(who, clip)
            if current is None:
                return error(409, UNREADABLE)
            try:
                annotation = current.apply(body["start"], body["end"], **fields)
            except ValueError as problem:
                return error(400, str(problem))
            store.save(who, clip_id, annotation)
        return jsonify(clip_payload(clip, annotation))

    @app.get("/sheet/<clip_id>/<int:sheet_index>")
    def sheet(clip_id, sheet_index):
        if current_annotator() is None:
            return error(401, "not signed in")
        sheets = index.sheets(clip_id) if is_safe_id(clip_id) else []
        match = next((s for s in sheets if s.sheet_index == sheet_index), None)
        if match is None:
            return error(404, "unknown sheet")
        if image_source is None:
            return error(501, "no image source configured")
        try:
            data = image_source(match.drive_file_id)
        except Exception:  # noqa: BLE001 - never leak Drive details to the browser
            log.exception("image source failed for sheet %s/%s", clip_id, sheet_index)
            return error(502, "image temporarily unavailable")
        return Response(data, mimetype="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})

    return app
