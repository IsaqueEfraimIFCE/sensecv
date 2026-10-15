"""Server wiring from environment variables (Fly.io secrets / env).

gunicorn entry point: ``frame_annotator.wsgi:create_app_from_env()`` (one worker).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

from .app import create_app
from .drive_images import CachedImageSource
from .drive_store import DriveAnnotationStore
from .index_db import IndexDb
from .labels import is_safe_id

MIN_TOKEN_LEN = 32
INDEX_NAME = "index.sqlite"


def _parse_tokens(raw) -> dict:
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


def _ensure_index(drive, sheets_folder: str, path: Path, force: bool) -> None:
    if path.exists() and not force:
        return
    file_id = drive.find(INDEX_NAME, sheets_folder)
    if file_id is None:
        raise RuntimeError(f"{INDEX_NAME} not found in the sheets folder; run the publisher first")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    with os.fdopen(fd, "wb") as handle:
        handle.write(drive.download(file_id))
    os.replace(tmp, path)


def build_app(env: Mapping, drive=None):
    tokens = _parse_tokens(env.get("ANNOTATOR_TOKENS_JSON"))
    sheets_folder = _required(env, "SHEETS_DRIVE_FOLDER_ID")
    annotations_folder = _required(env, "ANNOTATIONS_DRIVE_FOLDER_ID")
    if drive is None:
        from .drive import DriveClient, credentials_from_env

        drive = DriveClient(credentials_from_env(env))
    index_path = Path(env.get("ANNOTATOR_INDEX_DB", "/data/index.sqlite"))
    _ensure_index(drive, sheets_folder, index_path, force=env.get("ANNOTATOR_REFRESH_INDEX") == "1")
    cache_mb = int(env.get("IMAGE_CACHE_MAX_MB", "2000"))
    return create_app(
        index=IndexDb(index_path),
        store=DriveAnnotationStore(drive, annotations_folder),
        tokens=tokens,
        image_source=CachedImageSource(drive, env.get("IMAGE_CACHE_DIR", "/data/cache"),
                                       cache_mb * 1_000_000),
        secure_cookies=True,
    )


def create_app_from_env():
    return build_app(os.environ)
