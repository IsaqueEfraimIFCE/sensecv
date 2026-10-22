"""Server wiring from environment variables (Fly.io secrets / env).

gunicorn entry point: ``clip_annotator.wsgi:create_app_from_env()`` (one worker).
"""
from __future__ import annotations

import atexit
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

from .app import create_app
from .drive_store import DriveEventStore
from .events import is_safe_id
from .index_db import IndexDb
from .videos import CachedVideoSource

MIN_TOKEN_LEN = 32
INDEX_NAME = "index.sqlite"


def parse_tokens(raw) -> dict:
    try:
        tokens = json.loads(raw)
    except (TypeError, ValueError):
        raise ValueError("ANNOTATOR_TOKENS_JSON must be a JSON object of token -> name") from None
    if not isinstance(tokens, dict) or not tokens:
        raise ValueError("ANNOTATOR_TOKENS_JSON must name at least one annotator")
    for token, name in tokens.items():
        if len(token) < MIN_TOKEN_LEN:
            raise ValueError(f"annotator tokens must be at least {MIN_TOKEN_LEN} characters")
        if not is_safe_id(name):
            raise ValueError("annotator names may use letters, digits, '_', '.', '-' only")
    return tokens


def _required(env: Mapping, key: str) -> str:
    value = env.get(key)
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _ensure_index(drive, videos_folder: str, path: Path, force: bool) -> None:
    if path.exists() and not force:
        return
    file_id = drive.find(INDEX_NAME, videos_folder)
    if file_id is None:
        raise RuntimeError(f"{INDEX_NAME} not found in the videos folder; run the publisher first")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    with os.fdopen(fd, "wb") as handle:
        handle.write(drive.download(file_id))
    os.replace(tmp, path)


def build_app(env: Mapping, drive=None):
    tokens = parse_tokens(env.get("ANNOTATOR_TOKENS_JSON"))
    videos_folder = _required(env, "VIDEOS_DRIVE_FOLDER_ID")
    annotations_folder = _required(env, "ANNOTATIONS_DRIVE_FOLDER_ID")
    if drive is None:
        from .drive import DriveClient, credentials_from_env

        drive = DriveClient(credentials_from_env(env))
    index_path = Path(env.get("ANNOTATOR_INDEX_DB", "/data/index.sqlite"))
    _ensure_index(drive, videos_folder, index_path, force=env.get("ANNOTATOR_REFRESH_INDEX") == "1")
    cache_mb = int(env.get("VIDEO_CACHE_MAX_MB", "2000"))
    store = DriveEventStore(drive, annotations_folder, write_behind=True)
    atexit.register(store.close)  # flush unsaved work when the process exits
    return create_app(
        index=IndexDb(index_path),
        store=store,
        tokens=tokens,
        video_source=CachedVideoSource(drive, env.get("VIDEO_CACHE_DIR", "/data/cache"),
                                       cache_mb * 1_000_000),
        secure_cookies=True,
        client_ip_header=env.get("CLIENT_IP_HEADER", "Fly-Client-IP"),
    )


def create_app_from_env():
    return build_app(os.environ)
